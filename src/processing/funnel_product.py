"""
Étape 4 (correction) : entonnoir au niveau (client, produit).

Pourquoi : au niveau session, ~50 % des sessions d'achat n'ont pas d'événement 'cart'
(panier créé dans une autre session, ou achat direct). L'entonnoir par session n'est donc
pas monotone. Au niveau (client, produit) sur tout le mois, on suit un même produit
d'une session à l'autre.

Usage :
  docker exec spark-master /opt/spark/bin/spark-submit \
      --master "local[4]" --driver-memory 3g \
      /opt/project/src/processing/funnel_product.py
"""
import argparse
import csv
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/opt/project/data/silver/events")
    p.add_argument("--out-dir", default="/opt/project/docs/results")
    args = p.parse_args()

    spark = (
        SparkSession.builder.appName("funnel-product")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.time()

    ev = spark.read.parquet(args.input).select("user_id", "product_id", "event_type", "event_time")

    def first_time(t):
        return F.min(F.when(F.col("event_type") == t, F.col("event_time")))

    pairs = ev.groupBy("user_id", "product_id").agg(
        first_time("view").alias("t_view"),
        first_time("cart").alias("t_cart"),
        first_time("purchase").alias("t_purchase"),
    ).cache()

    has_view = F.col("t_view").isNotNull()
    has_cart = F.col("t_cart").isNotNull()
    has_purchase = F.col("t_purchase").isNotNull()
    bought_after_cart = has_cart & has_purchase & (F.col("t_purchase") >= F.col("t_cart"))

    def s(cond):
        return F.sum(F.when(cond, 1).otherwise(0))

    r = pairs.agg(
        F.count("*").alias("pairs_total"),
        s(has_view).alias("viewed"),
        s(has_view & has_cart).alias("viewed_carted"),
        s(has_view & bought_after_cart).alias("viewed_carted_purchased"),
        s(has_cart).alias("carted"),
        s(bought_after_cart).alias("carts_converted"),
        s(has_purchase).alias("purchased"),
        s(has_purchase & ~bought_after_cart).alias("purchases_without_prior_cart"),
    ).first()

    row = r.asDict()
    row["cart_abandon_rate"] = round(1 - row["carts_converted"] / row["carted"], 4)
    row["direct_purchase_share"] = round(row["purchases_without_prior_cart"] / row["purchased"], 4)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "funnel_product.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row.keys())
        w.writerow(row.values())

    print("\n" + "=" * 60)
    for k, v in row.items():
        print(f"{k:<32}{v:>15,}" if isinstance(v, int) else f"{k:<32}{v:>15}")
    print("=" * 60)
    print(f"-> {path}  ({time.time() - t0:.1f} s)")
    spark.stop()


if __name__ == "__main__":
    main()
