"""
Étape 2 : Ingestion CSV -> Parquet (couche Bronze).

Usage (dans le cluster Docker) :
  docker exec spark-master /opt/spark/bin/spark-submit \
      --master spark://spark-master:7077 \
      --driver-memory 2g --executor-memory 2g \
      /opt/project/src/ingestion/csv_to_parquet.py \
      --input /opt/project/data/raw/2019-Oct.csv \
      --output /opt/project/data/bronze/events

Test rapide sur 1 % des données : ajouter  --sample 0.01
"""
import argparse
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, LongType, StringType, StructField, StructType,
)

# Schéma explicite du dataset REES46 (évite un 2e passage pour inférer les types)
SCHEMA = StructType([
    StructField("event_time", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("product_id", LongType(), True),
    StructField("category_id", LongType(), True),
    StructField("category_code", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("user_id", LongType(), True),
    StructField("user_session", StringType(), True),
])


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Chemin du CSV (ou dossier de CSV)")
    p.add_argument("--output", required=True, help="Dossier de sortie Parquet")
    p.add_argument("--sample", type=float, default=None, help="Fraction à lire (ex: 0.01)")
    p.add_argument("--mode", default="overwrite", choices=["overwrite", "append"])
    args = p.parse_args()

    spark = (
        SparkSession.builder.appName("ingestion-csv-to-parquet")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    t0 = time.time()
    df = spark.read.csv(args.input, header=True, schema=SCHEMA)

    if args.sample:
        df = df.sample(fraction=args.sample, seed=42)

    # Format source : 2019-10-01 00:00:00 UTC
    df = (
        df.withColumn(
            "event_time",
            F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss 'UTC'"),
        )
        .withColumn("event_date", F.to_date("event_time"))
        .withColumn("ingestion_ts", F.current_timestamp())
        .withColumn("source_file", F.input_file_name())
    )

    # Partitionnement par date : accélère les requêtes filtrées sur une période
    (
        df.write.mode(args.mode)
        .partitionBy("event_date")
        .parquet(args.output)
    )
    t_write = time.time() - t0

    # Vérification : relecture et comptage
    t1 = time.time()
    out = spark.read.parquet(args.output)
    n = out.count()
    t_count = time.time() - t1

    print("\n" + "=" * 60)
    print(f"Lignes écrites       : {n:,}")
    print(f"Temps d'ingestion    : {t_write:.1f} s")
    print(f"Temps count Parquet  : {t_count:.1f} s")
    try:
        print(f"Taille Parquet       : {dir_size_mb(args.output):.0f} Mo")
        if os.path.isfile(args.input):
            print(f"Taille CSV source    : {os.path.getsize(args.input) / (1024 * 1024):.0f} Mo")
    except Exception as e:  # le chemin peut ne pas être visible depuis le driver
        print(f"(tailles non calculées : {e})")
    print("=" * 60)

    out.groupBy("event_type").count().orderBy(F.desc("count")).show()
    out.printSchema()

    spark.stop()


if __name__ == "__main__":
    main()
