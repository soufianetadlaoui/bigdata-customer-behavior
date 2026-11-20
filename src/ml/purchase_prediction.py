"""
Étape 6a : prédiction de l'achat dans les 10 jours suivants (Spark MLlib).

Usage :
  docker exec spark-master /opt/spark/bin/spark-submit \
      --master "local[4]" --driver-memory 3g \
      /opt/project/src/ml/purchase_prediction.py

Schéma temporel (sans fuite d'information) :
  Entraînement : variables = événements <= 2019-10-11, cible = achat du 12 au 21 octobre
  Test         : variables = événements <= 2019-10-21, cible = achat du 22 au 31 octobre
  La cible d'entraînement s'arrête à la date de coupure du test : le modèle n'a jamais
  accès aux achats de la période de test. Pas de validation croisée aléatoire (elle
  mélangerait passé et futur).

Population : clients ayant au moins un événement jusqu'à la date de coupure.
Les clients qui apparaissent après la coupure (démarrage à froid) ne sont pas évalués.

Le modèle final (--final-model) est choisi À L'AVANCE, pas d'après le score de test,
pour ne pas biaiser l'évaluation.
"""
import argparse
import csv
import datetime as dt
import os
import time

from pyspark.ml.classification import (
    GBTClassifier, LogisticRegression, RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

FEATURES = [
    "n_events", "n_views", "n_carts", "n_purchase_items", "n_sessions", "n_active_days",
    "n_products", "n_categories", "purchase_sessions", "monetary", "avg_view_price",
    "events_last3", "carts_last3", "sessions_last3", "events_last7",
    "recency_days", "tenure_days", "days_since_purchase", "has_purchased", "open_cart",
    "cart_rate", "events_per_session",
]


def write_rows(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))
    print(f"  -> {os.path.basename(path)} ({len(rows)} lignes)")


def build_snapshot(ev, cutoff, horizon):
    """Variables calculées sur event_date <= cutoff ; cible = achat dans (cutoff, cutoff+horizon]."""
    cut = F.lit(cutoff).cast("date")
    end = F.date_add(cut, horizon)
    cut_ts = F.unix_timestamp(F.lit(cutoff).cast("timestamp")) + 86400  # fin du jour de coupure

    past = ev.filter(F.col("event_date") <= cut).withColumn("ts", F.unix_timestamp("event_time"))
    et = F.col("event_type")

    def cnt(c):
        return F.sum(F.when(c, 1).otherwise(0))

    last3 = F.col("event_date") > F.date_sub(cut, 3)  # les 3 derniers jours, coupure incluse
    last7 = F.col("event_date") > F.date_sub(cut, 7)

    f = past.groupBy("user_id").agg(
        F.count("*").alias("n_events"),
        cnt(et == "view").alias("n_views"),
        cnt(et == "cart").alias("n_carts"),
        cnt(et == "purchase").alias("n_purchase_items"),
        F.countDistinct("user_session").alias("n_sessions"),
        F.countDistinct("event_date").alias("n_active_days"),
        # approximation (HyperLogLog) : évite un éclatement coûteux en mémoire
        F.approx_count_distinct("product_id").alias("n_products"),
        F.approx_count_distinct("category_id").alias("n_categories"),
        F.countDistinct(F.when(et == "purchase", F.col("user_session"))).alias("purchase_sessions"),
        F.sum(F.when(et == "purchase", F.col("price")).otherwise(0.0)).alias("monetary"),
        F.avg(F.when(et == "view", F.col("price"))).alias("avg_view_price"),
        cnt(last3).alias("events_last3"),
        cnt(last3 & (et == "cart")).alias("carts_last3"),
        F.countDistinct(F.when(last3, F.col("user_session"))).alias("sessions_last3"),
        cnt(last7).alias("events_last7"),
        F.max("ts").alias("last_ts"),
        F.min("ts").alias("first_ts"),
        F.max(F.when(et == "purchase", F.col("ts"))).alias("last_purchase_ts"),
        F.max(F.when(et == "cart", F.col("ts"))).alias("last_cart_ts"),
    )

    f = (
        f.withColumn("recency_days", (cut_ts - F.col("last_ts")) / 86400.0)
        .withColumn("tenure_days", (cut_ts - F.col("first_ts")) / 86400.0)
        .withColumn(
            "days_since_purchase",
            F.when(F.col("last_purchase_ts").isNull(), F.lit(99.0))
            .otherwise((cut_ts - F.col("last_purchase_ts")) / 86400.0),
        )
        .withColumn("has_purchased", F.col("last_purchase_ts").isNotNull().cast("double"))
        # panier ouvert : un ajout au panier plus récent que le dernier achat (ou aucun achat)
        .withColumn(
            "open_cart",
            (
                F.col("last_cart_ts").isNotNull()
                & (F.col("last_purchase_ts").isNull() | (F.col("last_cart_ts") > F.col("last_purchase_ts")))
            ).cast("double"),
        )
        .withColumn("cart_rate", F.col("n_carts") / (F.col("n_views") + 1.0))
        .withColumn("events_per_session", F.col("n_events") / F.col("n_sessions"))
        .fillna(0.0, subset=FEATURES)
    )

    pos = (
        ev.filter(
            (F.col("event_date") > cut) & (F.col("event_date") <= end) & (F.col("event_type") == "purchase")
        )
        .select("user_id").distinct().withColumn("label", F.lit(1.0))
    )
    return f.join(pos, "user_id", "left").fillna({"label": 0.0}).select("user_id", "label", *FEATURES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/opt/project/data/silver/events")
    p.add_argument("--segments", default="/opt/project/data/gold/customer_segments")
    p.add_argument("--scores-out", default="/opt/project/data/gold/purchase_scores")
    p.add_argument("--out-dir", default="/opt/project/docs/results")
    p.add_argument("--train-cutoff", default="2019-10-11")
    p.add_argument("--test-cutoff", default="2019-10-21")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--final-model", default="gbt", choices=["logistic_regression", "random_forest", "gbt"])
    args = p.parse_args()

    tr_c = dt.date.fromisoformat(args.train_cutoff)
    te_c = dt.date.fromisoformat(args.test_cutoff)
    if tr_c + dt.timedelta(days=args.horizon) > te_c:
        raise SystemExit("La cible d'entraînement chevauche la période de test : réduisez l'horizon.")

    spark = (
        SparkSession.builder.appName("purchase-prediction")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.time()

    ev = spark.read.parquet(args.input).select(
        "user_id", "user_session", "event_type", "event_time", "event_date",
        "price", "product_id", "category_id",
    )
    train = build_snapshot(ev, args.train_cutoff, args.horizon).cache()
    test = build_snapshot(ev, args.test_cutoff, args.horizon).cache()

    r = train.agg(F.count("*").alias("n"), F.sum("label").alias("p")).first()
    n_tr, pos_tr = r["n"], r["p"]
    r = test.agg(F.count("*").alias("n"), F.sum("label").alias("p")).first()
    n_te, pos_te = r["n"], r["p"]
    base_te = pos_te / n_te
    print(f"\nEntraînement : {n_tr:,} clients, {100 * pos_tr / n_tr:.2f} % achètent (coupure {args.train_cutoff})")
    print(f"Test         : {n_te:,} clients, {100 * base_te:.2f} % achètent (coupure {args.test_cutoff})\n")

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="keep")
    train_v = assembler.transform(train).select("user_id", "label", "features").cache()
    test_v = assembler.transform(test).select("user_id", "label", "features").cache()

    ev_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    ev_pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

    estimators = {
        "logistic_regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01),
        "random_forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=8, seed=42),
        "gbt": GBTClassifier(featuresCol="features", labelCol="label", maxIter=60, maxDepth=5, stepSize=0.1, seed=42),
    }

    metrics = []
    final_model, final_pred = None, None
    for name, est in estimators.items():
        t1 = time.time()
        model = est.fit(train_v)
        pred = (
            model.transform(test_v)
            .withColumn("prob", vector_to_array("probability")[1])
            .cache()
        )
        auc, aupr = ev_roc.evaluate(pred), ev_pr.evaluate(pred)

        # Ciblage : les 10 % de clients au score le plus élevé
        thr = pred.approxQuantile("prob", [0.9], 0.001)[0]
        top = pred.filter(F.col("prob") >= thr).agg(F.count("*").alias("n"), F.sum("label").alias("p")).first()
        precision = (top["p"] or 0.0) / top["n"]
        recall = (top["p"] or 0.0) / pos_te
        lift = precision / base_te
        metrics.append((name, round(auc, 4), round(aupr, 4), round(precision, 4),
                        round(recall, 4), round(lift, 2), n_tr, n_te,
                        round(pos_tr / n_tr, 4), round(base_te, 4)))
        print(f"{name:<22} AUC-ROC={auc:.4f}  AUC-PR={aupr:.4f}  "
              f"top10%: précision={precision:.3f} rappel={recall:.3f} lift={lift:.2f}  ({time.time() - t1:.0f} s)")
        if name == args.final_model:
            final_model, final_pred = model, pred

    print(f"\nTaux de base (test) : {100 * base_te:.2f} %  -> AUC-PR d'un modèle aléatoire = {base_te:.4f}")
    write_rows(
        os.path.join(args.out_dir, "purchase_model_metrics.csv"),
        ["model", "auc_roc", "auc_pr", "precision_top10", "recall_top10", "lift_top10",
         "train_rows", "test_rows", "train_base_rate", "test_base_rate"],
        metrics,
    )

    # Importance des variables (modèles à base d'arbres)
    if hasattr(final_model, "featureImportances"):
        imp = sorted(zip(FEATURES, final_model.featureImportances.toArray()), key=lambda x: -x[1])
        write_rows(os.path.join(args.out_dir, "purchase_feature_importance.csv"),
                   ["feature", "importance"], [(k, round(float(v), 5)) for k, v in imp])
        print("\nVariables les plus importantes :")
        for k, v in imp[:8]:
            print(f"  {k:<22}{v:.4f}")

    # Déciles de score (lift) du modèle final
    w = Window.orderBy(F.desc("prob"))
    dec = (
        final_pred.withColumn("decile", F.ntile(10).over(w))
        .groupBy("decile")
        .agg(F.count("*").alias("users"), F.sum("label").alias("buyers"),
             F.round(F.avg("label"), 4).alias("purchase_rate"))
        .orderBy("decile")
    )
    write_rows(os.path.join(args.out_dir, "purchase_deciles.csv"), dec.columns, dec.collect())

    # Scores individuels (pour le ciblage et le tableau de bord), avec le segment RFM si disponible
    scores = final_pred.select("user_id", "label", F.col("prob").alias("purchase_prob"))
    try:
        seg = spark.read.parquet(args.segments).select("user_id", "segment")
        scores = scores.join(seg, "user_id", "left")
    except Exception as e:  # la table de segments n'existe pas encore
        print(f"(segments non joints : {str(e)[:80]})")
    scores.write.mode("overwrite").parquet(args.scores_out)
    print(f"\nScores écrits : {args.scores_out}")
    print(f"Terminé en {time.time() - t0:.1f} s")
    spark.stop()


if __name__ == "__main__":
    main()
