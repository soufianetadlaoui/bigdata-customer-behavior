"""
Étape 7b : traitement en continu avec Spark Structured Streaming (source = dossier, sans Kafka).

Usage (le démarrer AVANT le producteur) :
  docker exec spark-master /opt/spark/bin/spark-submit \
      --master "local[4]" --driver-memory 3g \
      /opt/project/src/streaming/streaming_demo.py --reset

Deux traitements en parallèle :
  1. Indicateurs par fenêtre d'une minute (événements par type, chiffre d'affaires, clients actifs)
     -> data/stream/out/kpi_per_minute/
  2. Paniers abandonnés : jointure flux-flux "panier" x "achat du même produit par le même client
     dans les N minutes suivantes" (jointure externe gauche). Sans achat -> alerte, enrichie du
     segment RFM du client (jointure avec la table Gold statique).
     -> data/stream/out/abandoned_carts/  et  data/stream/out/cart_stats.csv

Temps d'événement : les fenêtres et les filigranes (watermarks) suivent l'heure des événements
(octobre 2019), pas l'heure de la machine. Une alerte apparaît donc quand le filigrane dépasse
l'heure du panier + N minutes + délai de tolérance.

Un panier "sans achat dans les N minutes" est une ALERTE (déclencheur de relance), pas un abandon
définitif : certains clients achètent plus tard.
"""
import argparse
import csv
import os
import shutil
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

SCHEMA = StructType([
    StructField("event_time", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("user_id", LongType(), True),
    StructField("user_session", StringType(), True),
    StructField("product_id", LongType(), True),
    StructField("category_code", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", DoubleType(), True),
])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/opt/project/data/stream")
    p.add_argument("--checkpoints", default="/tmp/stream_ckpt",
                   help="Hors du dossier partagé Windows : plus fiable pour l'état du flux")
    p.add_argument("--segments", default="/opt/project/data/gold/customer_segments")
    p.add_argument("--abandon-minutes", type=int, default=10)
    p.add_argument("--idle-timeout", type=int, default=90,
                   help="Arrêt automatique après N secondes sans nouvelles données")
    p.add_argument("--reset", action="store_true", help="Efface les sorties et les points de reprise")
    args = p.parse_args()

    incoming = f"{args.base}/incoming"
    out = f"{args.base}/out"
    kpi_dir, alerts_dir, stats_path = f"{out}/kpi_per_minute", f"{out}/abandoned_carts", f"{out}/cart_stats.csv"
    if args.reset:
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(args.checkpoints, ignore_errors=True)
    for d in (incoming, out):
        os.makedirs(d, exist_ok=True)

    spark = (
        SparkSession.builder.appName("streaming-demo")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")  # petit volume : 4 partitions d'état suffisent
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    seg = None
    try:
        seg = spark.read.parquet(args.segments).select("user_id", "segment").cache()
        print(f"Segments clients chargés : {seg.count():,}")
    except Exception as e:
        print(f"(segments indisponibles, alertes sans segment : {str(e)[:70]})")

    # --- Source : fichiers JSON déposés dans le dossier, un fichier par micro-lot -------------------
    events = (
        spark.readStream.schema(SCHEMA).option("maxFilesPerTrigger", 1).json(incoming)
        .withColumn("event_time", F.to_timestamp("event_time"))
    )

    # --- Traitement 1 : indicateurs par minute --------------------------------------------------------------
    kpi = (
        events.withWatermark("event_time", "1 minute")
        .groupBy(F.window("event_time", "1 minute").alias("w"), "event_type")
        .agg(
            F.count("*").alias("events"),
            F.round(F.sum(F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0.0)), 2).alias("revenue"),
            F.approx_count_distinct("user_id").alias("users"),
        )
        .select(F.col("w.start").alias("window_start"), "event_type", "events", "revenue", "users")
    )
    q_kpi = (
        kpi.writeStream.queryName("kpi_par_minute").outputMode("append").format("json")
        .option("path", kpi_dir).option("checkpointLocation", f"{args.checkpoints}/kpi")
        .trigger(processingTime="1 second").start()
    )

    # --- Traitement 2 : paniers sans achat dans les N minutes ------------------------------------------------------
    carts = (
        events.filter(F.col("event_type") == "cart")
        .select(F.col("user_id").alias("c_user"), F.col("product_id").alias("c_product"),
                F.col("event_time").alias("cart_time"), "price", "brand", "category_code")
        .withWatermark("cart_time", "1 minute")
    )
    purchases = (
        events.filter(F.col("event_type") == "purchase")
        .select(F.col("user_id").alias("p_user"), F.col("product_id").alias("p_product"),
                F.col("event_time").alias("purchase_time"))
        .withWatermark("purchase_time", "1 minute")
    )
    joined = carts.join(
        purchases,
        F.expr(f"c_user = p_user AND c_product = p_product AND purchase_time >= cart_time "
               f"AND purchase_time <= cart_time + interval {args.abandon_minutes} minutes"),
        "leftOuter",
    )

    def handle(batch_df, batch_id):
        b = batch_df.persist()
        r = b.agg(
            F.sum(F.when(F.col("purchase_time").isNull(), 1).otherwise(0)).alias("a"),
            F.sum(F.when(F.col("purchase_time").isNotNull(), 1).otherwise(0)).alias("c"),
        ).first()
        a, c = int(r["a"] or 0), int(r["c"] or 0)
        if a or c:
            if a:
                ab = b.filter(F.col("purchase_time").isNull()).select(
                    F.col("c_user").alias("user_id"), F.col("c_product").alias("product_id"),
                    "brand", "category_code", "price", "cart_time",
                )
                if seg is not None:
                    ab = ab.join(F.broadcast(seg), "user_id", "left")
                ab.write.mode("append").json(alerts_dir)
            new_file = not os.path.exists(stats_path)
            with open(stats_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["batch_id", "converted", "abandoned"])
                w.writerow([batch_id, c, a])
            print(f"  [lot {batch_id}] paniers convertis : {c} | alertes d'abandon : {a}", flush=True)
        b.unpersist()

    q_cart = (
        joined.writeStream.queryName("paniers_abandonnes").outputMode("append")
        .foreachBatch(handle).option("checkpointLocation", f"{args.checkpoints}/carts")
        .trigger(processingTime="1 second").start()
    )

    print(f"\nFlux démarré. En attente de fichiers dans {incoming}")
    print("Lancez maintenant le producteur (autre terminal). Ctrl+C pour arrêter.\n")

    # --- Surveillance ------------------------------------------------------------------------------------------------------
    queries = [q_kpi, q_cart]
    t0 = last_activity = time.time()
    last_print = 0.0
    seen = {}
    try:
        while all(q.isActive for q in queries):
            time.sleep(2)
            for q in queries:
                lp = q.lastProgress
                if lp and lp.get("numInputRows", 0) > 0 and seen.get(q.name) != lp.get("batchId"):
                    seen[q.name] = lp.get("batchId")
                    last_activity = time.time()
            if time.time() - last_print >= 10:
                last_print = time.time()
                for q in queries:
                    lp = q.lastProgress
                    if lp:
                        wm = (lp.get("eventTime") or {}).get("watermark", "-")
                        print(f"  [{int(time.time() - t0)} s] {q.name} : lot {lp.get('batchId')}, filigrane {wm}", flush=True)
            if time.time() - last_activity > args.idle_timeout:
                print(f"\nAucune donnée nouvelle depuis {args.idle_timeout} s : arrêt.")
                break
    except KeyboardInterrupt:
        print("\nInterruption demandée.")

    for q in queries:
        if q.exception():
            print(f"ERREUR dans {q.name} : {q.exception()}")
        if q.isActive:
            q.stop()

    # --- Bilan ----------------------------------------------------------------------------------------------------------------------
    conv = aband = 0
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                conv += int(row["converted"])
                aband += int(row["abandoned"])
    print("\n" + "=" * 62)
    print(f"Paniers convertis dans les {args.abandon_minutes} min : {conv:,}")
    print(f"Alertes d'abandon                : {aband:,}")
    if conv + aband:
        print(f"Part sans achat à {args.abandon_minutes} min          : {100 * aband / (conv + aband):.1f} %")
    print("=" * 62)
    try:
        spark.read.json(alerts_dir).groupBy("segment").count().orderBy(F.desc("count")).show(truncate=False)
    except Exception:
        pass
    spark.stop()


if __name__ == "__main__":
    main()
