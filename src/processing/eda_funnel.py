"""
Étape 4 : analyse exploratoire et entonnoir de conversion (Silver -> tables agrégées).

Usage :
  docker exec spark-master /opt/spark/bin/spark-submit \
      --master "local[4]" --driver-memory 3g \
      /opt/project/src/processing/eda_funnel.py

Produit de petits CSV dans docs/results/ (versionnables), lus ensuite par make_charts.py.

Définitions :
  - Session avec vue / panier / achat : au moins un événement de ce type dans la session.
  - Entonnoir : sessions -> sessions avec panier -> sessions avec achat.
  - Taux d'abandon de panier : 1 - (sessions panier ET achat) / (sessions avec panier).
"""
import argparse
import csv
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def save(df, name, out_dir):
    """Collecte un petit DataFrame agrégé et l'écrit en CSV (sans dépendance pandas)."""
    rows = df.collect()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(df.columns)
        for r in rows:
            w.writerow(list(r))
    print(f"  -> {name}.csv ({len(rows)} lignes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/opt/project/data/silver/events")
    p.add_argument("--out-dir", default="/opt/project/docs/results")
    args = p.parse_args()

    spark = (
        SparkSession.builder.appName("eda-funnel")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.time()

    price = F.col("price")
    ev = (
        spark.read.parquet(args.input)
        .withColumn(
            "price_band",
            F.when(price < 20, "1_<20")
            .when(price < 50, "2_20-50")
            .when(price < 100, "3_50-100")
            .when(price < 250, "4_100-250")
            .when(price < 500, "5_250-500")
            .otherwise("6_500+"),
        )
        .cache()
    )
    print(f"Événements Silver : {ev.count():,}")

    def cnt(t):
        return F.sum(F.when(F.col("event_type") == t, 1).otherwise(0))

    def flag(t):
        return F.max((F.col("event_type") == t).cast("int"))

    def buyer_revenue():
        return F.sum(F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0.0))

    # 1. KPI globaux ------------------------------------------------------------
    kpis = ev.agg(
        F.count("*").alias("events"),
        F.countDistinct("user_id").alias("users"),
        F.countDistinct("user_session").alias("sessions"),
        F.countDistinct("product_id").alias("products"),
        F.countDistinct("brand").alias("brands"),
        cnt("purchase").alias("purchases"),
        F.round(buyer_revenue(), 2).alias("revenue"),
        F.countDistinct(F.when(F.col("event_type") == "purchase", F.col("user_id"))).alias("buyers"),
    )
    save(kpis, "kpis", args.out_dir)

    # 2. Évolution quotidienne ----------------------------------------------------
    daily = (
        ev.groupBy("event_date")
        .agg(
            cnt("view").alias("views"),
            cnt("cart").alias("carts"),
            cnt("purchase").alias("purchases"),
            F.round(buyer_revenue(), 2).alias("revenue"),
            F.countDistinct("user_id").alias("users"),
        )
        .orderBy("event_date")
    )
    save(daily, "daily", args.out_dir)

    # 3. Activité par heure et jour de la semaine (1 = dimanche) -------------------
    hourly = (
        ev.groupBy("day_of_week", "hour")
        .agg(F.count("*").alias("events"), cnt("purchase").alias("purchases"))
        .orderBy("day_of_week", "hour")
    )
    save(hourly, "hourly", args.out_dir)

    # 4. Entonnoir de conversion au niveau session ----------------------------------
    sess = (
        ev.groupBy("user_session")
        .agg(
            flag("view").alias("v"),
            flag("cart").alias("c"),
            flag("purchase").alias("p"),
            F.min(F.when(F.col("event_type") == "cart", F.col("event_time"))).alias("first_cart"),
            F.min(F.when(F.col("event_type") == "purchase", F.col("event_time"))).alias("first_purchase"),
        )
        .cache()
    )
    funnel = sess.agg(
        F.count("*").alias("sessions"),
        F.sum("v").alias("with_view"),
        F.sum("c").alias("with_cart"),
        F.sum("p").alias("with_purchase"),
        F.sum(F.col("c") * F.col("p")).alias("cart_and_purchase"),
    ).withColumn("cart_abandon_rate", F.round(1 - F.col("cart_and_purchase") / F.col("with_cart"), 4))
    save(funnel, "funnel", args.out_dir)

    # 5. Temps entre le premier ajout au panier et l'achat (minutes) ------------------
    t = (
        sess.filter(
            F.col("first_cart").isNotNull()
            & F.col("first_purchase").isNotNull()
            & (F.col("first_purchase") >= F.col("first_cart"))
        )
        .withColumn(
            "minutes",
            (F.unix_timestamp("first_purchase") - F.unix_timestamp("first_cart")) / 60,
        )
    )
    probs = [0.25, 0.5, 0.75, 0.9]
    qs = t.approxQuantile("minutes", probs, 0.01)
    if len(qs) == len(probs):
        save(spark.createDataFrame(list(zip(probs, [float(x) for x in qs])), ["quantile", "minutes"]),
             "time_to_purchase", args.out_dir)

    # 6. Entonnoir par dimension (catégorie, tranche de prix) ----------------------------
    def funnel_by(dim):
        s = ev.groupBy("user_session", dim).agg(
            flag("view").alias("v"), flag("cart").alias("c"), flag("purchase").alias("p")
        )
        return (
            s.groupBy(dim)
            .agg(
                F.count("*").alias("sessions"),
                F.sum("c").alias("carted"),
                F.sum("p").alias("purchased"),
                F.sum(F.col("c") * F.col("p")).alias("cart_and_purchase"),
            )
            .withColumn("cart_rate", F.round(F.col("carted") / F.col("sessions"), 4))
            .withColumn("purchase_rate", F.round(F.col("purchased") / F.col("sessions"), 4))
            .withColumn("cart_to_purchase", F.round(F.col("cart_and_purchase") / F.col("carted"), 4))
            .orderBy(F.desc("sessions"))
        )

    save(funnel_by("category_main"), "funnel_category", args.out_dir)
    save(funnel_by("price_band").orderBy("price_band"), "funnel_price_band", args.out_dir)

    # 7. Marques et produits ------------------------------------------------------------
    brands = (
        ev.filter(F.col("brand") != "unknown")
        .groupBy("brand")
        .agg(
            cnt("view").alias("views"),
            cnt("cart").alias("carts"),
            cnt("purchase").alias("purchases"),
            F.round(buyer_revenue(), 2).alias("revenue"),
        )
        .orderBy(F.desc("revenue"))
        .limit(15)
    )
    save(brands, "top_brands", args.out_dir)

    products = (
        ev.groupBy("product_id")
        .agg(
            F.first("brand").alias("brand"),
            F.first("category_main").alias("category_main"),
            cnt("view").alias("views"),
            cnt("cart").alias("carts"),
            cnt("purchase").alias("purchases"),
        )
        .orderBy(F.desc("views"))
        .limit(20)
    )
    save(products, "top_products", args.out_dir)

    # 8. Cohortes hebdomadaires (rétention) ------------------------------------------------
    d0 = ev.agg(F.min("event_date")).first()[0]
    wk = ev.select(
        "user_id", F.floor(F.datediff("event_date", F.lit(d0)) / 7).alias("week")
    ).distinct()
    cohort = wk.groupBy("user_id").agg(F.min("week").alias("cohort_week"))
    coh = (
        wk.join(cohort, "user_id")
        .groupBy("cohort_week", "week")
        .agg(F.count("*").alias("users"))
        .orderBy("cohort_week", "week")
    )
    save(coh, "cohorts", args.out_dir)

    # 9. Clients récurrents (nombre d'achats par acheteur) ---------------------------------------
    rep = (
        ev.filter(F.col("event_type") == "purchase")
        .groupBy("user_id")
        .count()
        .withColumn(
            "purchases",
            F.when(F.col("count") >= 5, F.lit("5+")).otherwise(F.col("count").cast("string")),
        )
        .groupBy("purchases")
        .agg(F.count("*").alias("buyers"))
        .orderBy("purchases")
    )
    save(rep, "repeat_buyers", args.out_dir)

    print(f"\nTerminé en {time.time() - t0:.1f} s. CSV dans {args.out_dir}")
    spark.stop()


if __name__ == "__main__":
    main()
