"""
Étape 5 : segmentation client RFM + K-Means (Spark MLlib).

Usage :
  docker exec spark-master /opt/spark/bin/spark-submit \
      --master "local[4]" --driver-memory 3g \
      /opt/project/src/ml/rfm_segmentation.py            # k choisi automatiquement
  ... rfm_segmentation.py --k 4                           # k imposé
  ... rfm_segmentation.py --k 3 --names "Champions,Occasionnels,Faible valeur"
        (noms appliqués dans l'ordre du chiffre d'affaires décroissant)

Méthode :
  - Acheteurs : variables RFM + comportement, log-transformées puis standardisées, K-Means.
      Récence    = jours entre le dernier achat et la fin de la période
      Fréquence  = nombre de SESSIONS d'achat distinctes (pas d'articles)
      Montant    = somme des prix des articles achetés
      + nombre de sessions et diversité des produits (catégories fines, category_id)
  - k : choisi parmi 3..6 par silhouette maximale (ou imposé avec --k).
  - Non-acheteurs : règle simple (panier sans achat / simples visiteurs).
  - Segments numérotés par chiffre d'affaires décroissant ; à nommer d'après le profil (--names).
"""
import argparse
import csv
import os
import time

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def write_rows(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))
    print(f"  -> {os.path.basename(path)} ({len(rows)} lignes)")


def save_df(df, name, out_dir):
    write_rows(os.path.join(out_dir, f"{name}.csv"), df.columns, df.collect())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/opt/project/data/silver/events")
    p.add_argument("--gold", default="/opt/project/data/gold/customer_segments")
    p.add_argument("--out-dir", default="/opt/project/docs/results")
    p.add_argument("--k", type=int, default=None, help="Nombre de clusters (sinon choix auto 3..6)")
    p.add_argument("--names", default=None,
                   help="Noms des segments séparés par des virgules, du plus au moins rentable")
    args = p.parse_args()

    spark = (
        SparkSession.builder.appName("rfm-segmentation")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.time()

    ev = spark.read.parquet(args.input).select(
        "user_id", "user_session", "event_type", "event_time", "price", "category_id"
    )
    snap = ev.agg(F.max("event_time").alias("snap"))  # fin de la période observée

    # --- Variables par client --------------------------------------------------------
    is_p = F.col("event_type") == "purchase"
    users = (
        ev.groupBy("user_id")
        .agg(
            F.count(F.when(F.col("event_type") == "view", 1)).alias("n_views"),
            F.count(F.when(F.col("event_type") == "cart", 1)).alias("n_carts"),
            F.count(F.when(is_p, 1)).alias("n_purchase_items"),
            F.countDistinct(F.when(is_p, F.col("user_session"))).alias("frequency"),
            F.sum(F.when(is_p, F.col("price")).otherwise(0.0)).alias("monetary"),
            F.max(F.when(is_p, F.col("event_time"))).alias("last_purchase"),
            F.countDistinct("user_session").alias("n_sessions"),
            F.countDistinct(F.when(is_p, F.col("category_id"))).alias("n_categories"),
        )
        .crossJoin(snap)
        .withColumn(
            "recency_days",
            (F.unix_timestamp("snap") - F.unix_timestamp("last_purchase")) / 86400.0,
        )
        .cache()
    )
    buyers = users.filter(F.col("n_purchase_items") > 0)
    non_buyers = users.filter(F.col("n_purchase_items") == 0)
    n_buyers = buyers.count()
    print(f"Clients : {users.count():,} (acheteurs : {n_buyers:,})")

    # --- Préparation : log (variables asymétriques) puis standardisation -----------------
    feat_cols = ["f_recency", "f_frequency", "f_monetary", "f_sessions", "f_categories"]
    feat = (
        buyers.withColumn("f_recency", F.col("recency_days"))
        .withColumn("f_frequency", F.log1p("frequency"))
        .withColumn("f_monetary", F.log1p("monetary"))
        .withColumn("f_sessions", F.log1p("n_sessions"))
        .withColumn("f_categories", F.log1p("n_categories"))
    )
    vec = VectorAssembler(inputCols=feat_cols, outputCol="raw_features").transform(feat)
    scaled = (
        StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
        .fit(vec)
        .transform(vec)
        .cache()
    )

    # --- Choix de k : coude (WSSSE) et silhouette -----------------------------------------
    evaluator = ClusteringEvaluator(
        featuresCol="features", predictionCol="cluster",
        metricName="silhouette", distanceMeasure="squaredEuclidean",
    )
    scores = []
    for k in range(2, 9):
        model = KMeans(k=k, seed=42, maxIter=50, featuresCol="features", predictionCol="cluster").fit(scaled)
        sil = evaluator.evaluate(model.transform(scaled))
        scores.append((k, float(model.summary.trainingCost), float(sil)))
        print(f"  k={k}  WSSSE={scores[-1][1]:,.0f}  silhouette={sil:.4f}")

    if args.k:
        best_k = args.k
    else:
        best_k = max((s for s in scores if 3 <= s[0] <= 6), key=lambda s: s[2])[0]
    print(f"\nk retenu : {best_k}" + (" (imposé)" if args.k else " (silhouette max entre 3 et 6)"))
    write_rows(
        os.path.join(args.out_dir, "k_selection.csv"),
        ["k", "wssse", "silhouette", "chosen"],
        [(k, w, s, int(k == best_k)) for k, w, s in scores],
    )

    final = KMeans(k=best_k, seed=42, maxIter=100, featuresCol="features", predictionCol="cluster").fit(scaled)
    clustered = final.transform(scaled)

    # --- Profil des clusters et noms suggérés -------------------------------------------------
    prof = clustered.groupBy("cluster").agg(
        F.count("*").alias("users"),
        F.round(F.avg("recency_days"), 2).alias("avg_recency_days"),
        F.round(F.avg("frequency"), 2).alias("avg_frequency"),
        F.round(F.avg("monetary"), 2).alias("avg_monetary"),
        F.round(F.avg("n_sessions"), 2).alias("avg_sessions"),
        F.round(F.avg("n_categories"), 2).alias("avg_categories"),
        F.round(F.sum("monetary"), 2).alias("revenue"),
    )
    rows = prof.orderBy(F.desc("revenue")).collect()  # du plus au moins rentable
    custom = [n.strip() for n in args.names.split(",")] if args.names else []
    if custom and len(custom) != len(rows):
        raise SystemExit(f"--names doit contenir {len(rows)} noms (reçu {len(custom)})")
    labels = custom or [f"Segment {i + 1}" for i in range(len(rows))]
    mapping = {int(r["cluster"]): labels[i] for i, r in enumerate(rows)}
    names = spark.createDataFrame(list(mapping.items()), ["cluster", "segment"])

    prof_named = prof.join(names, "cluster").orderBy(F.desc("revenue"))
    prof_named = prof_named.select(
        "cluster", "segment", "users", "avg_recency_days", "avg_frequency",
        "avg_monetary", "avg_sessions", "avg_categories", "revenue",
    )
    print("\nProfil des segments d'acheteurs (Segment 1 = le plus rentable) :")
    prof_named.show(truncate=False)
    save_df(prof_named, "segment_profiles", args.out_dir)

    # --- Table finale : tous les clients ------------------------------------------------------------
    cols = ["user_id", "segment", "segment_type", "cluster", "recency_days",
            "frequency", "monetary", "n_sessions", "n_views", "n_carts"]
    seg_buyers = clustered.join(names, "cluster").withColumn("segment_type", F.lit("buyer")).select(*cols)
    seg_non = (
        non_buyers.withColumn(
            "segment",
            F.when(F.col("n_carts") > 0, "Panier sans achat").otherwise("Visiteurs"),
        )
        .withColumn("segment_type", F.lit("non_buyer"))
        .withColumn("cluster", F.lit(None).cast("int"))
        .select(*cols)
    )
    segments = seg_buyers.unionByName(seg_non)
    segments.write.mode("overwrite").parquet(args.gold)

    total_users = segments.count()
    total_rev = segments.agg(F.sum("monetary")).first()[0]
    sizes = (
        segments.groupBy("segment", "segment_type")
        .agg(F.count("*").alias("users"), F.round(F.sum("monetary"), 2).alias("revenue"))
        .withColumn("user_share", F.round(F.col("users") / total_users, 4))
        .withColumn("revenue_share", F.round(F.col("revenue") / F.lit(total_rev), 4))
        .orderBy(F.desc("revenue"), F.desc("users"))
    )
    print("Taille et chiffre d'affaires par segment :")
    sizes.show(truncate=False)
    save_df(sizes, "segment_sizes", args.out_dir)

    # Échantillon pour le nuage de points
    frac = min(1.0, 4000.0 / max(n_buyers, 1))
    sample = (
        seg_buyers.sample(False, frac, seed=42)
        .select("segment", "recency_days", "frequency", "monetary")
    )
    save_df(sample, "segment_sample", args.out_dir)

    print(f"\nTerminé en {time.time() - t0:.1f} s. Table Gold : {args.gold}")
    spark.stop()


if __name__ == "__main__":
    main()
