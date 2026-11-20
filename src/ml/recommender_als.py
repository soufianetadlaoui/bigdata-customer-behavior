"""
Étape 6b : recommandation de produits (ALS, feedback implicite) avec Spark MLlib.

Usage :
  docker exec spark-master /opt/spark/bin/spark-submit \
      --master "local[4]" --driver-memory 3g \
      /opt/project/src/ml/recommender_als.py

Évaluation temporelle (sans fuite) :
  Entraînement : interactions jusqu'au 2019-10-21
                 poids : vue = 1, panier = 3, achat = 5 ; note = log1p(somme des poids)
  Vérité       : produits ACHETÉS du 22 au 31 octobre par des clients connus à l'entraînement
  Métriques    : Precision@10, Recall@10, HitRate@10
  Références   : (1) produits les plus achetés (même liste pour tous les clients)
                 (2) produits récemment consultés par le client (reciblage)

Remarque : les produits déjà vus ne sont pas exclus des recommandations, car un client
achète souvent un produit qu'il a déjà consulté ; c'est aussi ce que fait la référence (2).
"""
import argparse
import csv
import os
import time

from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

K = 10
INT_MAX = 2**31 - 1


def write_rows(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))
    print(f"  -> {os.path.basename(path)} ({len(rows)} lignes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/opt/project/data/silver/events")
    p.add_argument("--gold", default="/opt/project/data/gold")
    p.add_argument("--out-dir", default="/opt/project/docs/results")
    p.add_argument("--cutoff", default="2019-10-21")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--sample-users", type=int, default=20000,
                   help="Nombre de clients supplémentaires dont on sauvegarde les recommandations")
    args = p.parse_args()

    spark = (
        SparkSession.builder.appName("recommender-als")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.time()

    ev = spark.read.parquet(args.input).select(
        "user_id", "product_id", "event_type", "event_time", "event_date",
        "brand", "category_code", "category_main", "price",
    )
    mx = ev.agg(F.max("user_id").alias("u"), F.max("product_id").alias("p")).first()
    if mx["u"] > INT_MAX or mx["p"] > INT_MAX:
        raise SystemExit("Les identifiants dépassent la capacité d'ALS (entiers 32 bits).")

    cut = F.lit(args.cutoff).cast("date")
    end = F.date_add(cut, args.horizon)
    et = F.col("event_type")
    train_ev = ev.filter(F.col("event_date") <= cut)
    test_buys = ev.filter((F.col("event_date") > cut) & (F.col("event_date") <= end) & (et == "purchase"))

    uid = F.col("user_id").cast("int").alias("user_id")
    pid = F.col("product_id").cast("int").alias("product_id")

    # --- Matrice d'interactions implicites ---------------------------------------------------
    weight = F.when(et == "view", 1.0).when(et == "cart", 3.0).when(et == "purchase", 5.0).otherwise(0.0)
    inter = (
        train_ev.groupBy(uid, pid)
        .agg(F.sum(weight).alias("w"))
        .filter(F.col("w") > 0)
        .withColumn("rating", F.log1p("w").cast("float"))
        .select("user_id", "product_id", "rating")
        .cache()
    )
    n_inter = inter.count()
    train_users = inter.select("user_id").distinct().cache()
    n_users = train_users.count()
    n_items = inter.select("product_id").distinct().count()
    print(f"\nInteractions : {n_inter:,} | clients : {n_users:,} | produits : {n_items:,}")
    print(f"Densité de la matrice : {100 * n_inter / (n_users * n_items):.4f} %")

    # --- Vérité terrain : achats futurs de clients connus ------------------------------------------
    truth = (
        test_buys.select(uid, pid).distinct()
        .groupBy("user_id").agg(F.collect_set("product_id").alias("truth"))
    )
    eval_truth = truth.join(train_users, "user_id").cache()
    n_eval = eval_truth.count()
    print(f"Clients évalués (connus, avec achat du {args.cutoff}+1 à +{args.horizon} j) : {n_eval:,}\n")

    # --- Modèle ALS --------------------------------------------------------------------------------------
    t1 = time.time()
    als = ALS(
        userCol="user_id", itemCol="product_id", ratingCol="rating", implicitPrefs=True,
        rank=args.rank, maxIter=args.iters, regParam=args.reg, alpha=args.alpha,
        coldStartStrategy="drop", seed=42,
    )
    model = als.fit(inter)
    print(f"ALS entraîné en {time.time() - t1:.0f} s (rang={args.rank}, reg={args.reg}, alpha={args.alpha})")
    recs = model.recommendForUserSubset(eval_truth.select("user_id"), K).select(
        "user_id", F.col("recommendations.product_id").alias("rec_items")
    )

    # --- Références ---------------------------------------------------------------------------------------------
    pop = [
        r["product_id"]
        for r in train_ev.filter(et == "purchase").groupBy(pid).count()
        .orderBy(F.desc("count")).limit(K).collect()
    ]
    pop_recs = eval_truth.select("user_id").withColumn(
        "rec_items", F.array(*[F.lit(int(x)) for x in pop])
    )

    last = (
        train_ev.filter(et.isin("view", "cart", "purchase"))
        .groupBy(uid, pid).agg(F.max("event_time").alias("t"))
    )
    w = Window.partitionBy("user_id").orderBy(F.desc("t"))
    recent_recs = (
        last.join(eval_truth.select("user_id"), "user_id")
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") <= K)
        .groupBy("user_id").agg(F.collect_list("product_id").alias("rec_items"))
    )

    # --- Métriques ---------------------------------------------------------------------------------------------------
    def evaluate(name, rec_df):
        j = rec_df.join(eval_truth, "user_id").withColumn(
            "hits", F.size(F.array_intersect("rec_items", "truth"))
        )
        r = j.agg(
            F.count("*").alias("n"),
            F.avg(F.col("hits") / F.lit(float(K))).alias("p"),
            F.avg(F.col("hits") / F.size("truth")).alias("r"),
            F.avg((F.col("hits") > 0).cast("double")).alias("h"),
        ).first()
        return (name, round(r["p"], 4), round(r["r"], 4), round(r["h"], 4), r["n"])

    rows = [
        evaluate("popularite_achats", pop_recs),
        evaluate("produits_recents", recent_recs),
        evaluate("als", recs),
    ]
    print(f"\n{'Méthode':<22}{'P@10':>9}{'R@10':>9}{'HitRate@10':>12}{'Clients':>10}")
    for n, pr, rc, hr, cnt in rows:
        print(f"{n:<22}{pr:>9.4f}{rc:>9.4f}{hr:>12.4f}{cnt:>10,}")
    write_rows(
        os.path.join(args.out_dir, "recommender_metrics.csv"),
        ["model", "precision_at_10", "recall_at_10", "hit_rate_at_10", "users_evaluated"], rows,
    )

    # --- Catalogue produits + recommandations pour le tableau de bord ---------------------------------------------------
    catalog = ev.groupBy(pid).agg(
        F.first("brand").alias("brand"),
        F.first("category_code").alias("category_code"),
        F.first("category_main").alias("category_main"),
        F.round(F.avg("price"), 2).alias("avg_price"),
    )
    catalog.write.mode("overwrite").parquet(os.path.join(args.gold, "product_catalog"))

    frac = min(1.0, args.sample_users / max(n_users, 1))
    rec_users = train_users.sample(False, frac, seed=42).unionByName(eval_truth.select("user_id")).distinct()
    flat = (
        model.recommendForUserSubset(rec_users, K)
        .select("user_id", F.posexplode("recommendations").alias("pos", "rec"))
        .select(
            "user_id", (F.col("pos") + 1).alias("rank"),
            F.col("rec.product_id").alias("product_id"), F.col("rec.rating").alias("score"),
        )
    )
    flat.write.mode("overwrite").parquet(os.path.join(args.gold, "recommendations"))
    print(f"\nRecommandations écrites pour {rec_users.count():,} clients dans {args.gold}/recommendations")

    # --- Exemples qualitatifs pour le rapport ---------------------------------------------------------------------------------
    ex = (
        recs.join(eval_truth, "user_id")
        .withColumn("hits", F.size(F.array_intersect("rec_items", "truth")))
        .filter("hits > 0").limit(3).collect()
    )
    ids = set()
    for r in ex:
        ids.update(r["rec_items"][:5])
        ids.update(r["truth"][:5])
    cat = {r["product_id"]: r for r in catalog.filter(F.col("product_id").isin(list(ids))).collect()} if ids else {}

    def describe(i, hit=False):
        c = cat.get(i)
        d = f"{c['brand']} | {c['category_code'] or '?'} | ~{c['avg_price']}" if c else "?"
        return f"    {'*' if hit else ' '} {i}  {d}"

    print("\nExemples (* = produit effectivement acheté ensuite) :")
    for r in ex:
        truth_set = set(r["truth"])
        print(f"  Client {r['user_id']} - achats réels : {len(r['truth'])}")
        for i in r["rec_items"][:5]:
            print(describe(i, i in truth_set))

    print(f"\nTerminé en {time.time() - t0:.1f} s")
    spark.stop()


if __name__ == "__main__":
    main()
