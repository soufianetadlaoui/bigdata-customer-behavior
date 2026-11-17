"""
Étape 3 : Bronze -> Silver (nettoyage, validation, enrichissement).

Usage :
  docker exec spark-master /opt/spark/bin/spark-submit \
      --master "local[4]" --driver-memory 3g \
      /opt/project/src/processing/bronze_to_silver.py

Règles de qualité appliquées (dans cet ordre) :
  R1  champs obligatoires nuls (event_time, event_type, user_id, product_id)
  R2  type d'événement invalide
  R3  prix nul, négatif ou manquant
  R4  user_session manquante
  R5  doublons exacts (event_time, event_type, product_id, user_id, user_session)
  R6  sessions aberrantes (probables bots) : plus de N événements par session
"""
import argparse
import json
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

VALID_EVENTS = ["view", "cart", "remove_from_cart", "purchase"]
DEDUP_KEYS = ["event_time", "event_type", "product_id", "user_id", "user_session"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/opt/project/data/bronze/events")
    p.add_argument("--output", default="/opt/project/data/silver/events")
    p.add_argument("--rejected", default="/opt/project/data/silver/rejected")
    p.add_argument("--report", default="/opt/project/docs/quality_report.json")
    p.add_argument("--max-session-events", type=int, default=500,
                   help="Seuil R6 : au-delà, la session est considérée comme un bot")
    args = p.parse_args()

    spark = (
        SparkSession.builder.appName("bronze-to-silver")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.time()

    bronze = spark.read.parquet(args.input).drop("source_file", "ingestion_ts").cache()
    n_raw = bronze.count()

    # --- R1 à R4 : règles de validation par ligne (une seule passe) ---------
    reason = (
        F.when(
            F.col("event_time").isNull() | F.col("event_type").isNull()
            | F.col("user_id").isNull() | F.col("product_id").isNull(),
            "R1_champs_obligatoires_nuls",
        )
        .when(~F.col("event_type").isin(VALID_EVENTS), "R2_type_evenement_invalide")
        .when(F.col("price").isNull() | (F.col("price") <= 0), "R3_prix_invalide")
        .when(F.col("user_session").isNull(), "R4_session_nulle")
    )
    flagged = bronze.withColumn("reject_reason", reason)

    rule_counts = {
        r["reject_reason"]: r["count"]
        for r in flagged.filter(F.col("reject_reason").isNotNull())
        .groupBy("reject_reason").count().collect()
    }
    n_rejected_rows = sum(rule_counts.values())

    # Les lignes rejetées sont conservées pour audit
    rejected = flagged.filter(F.col("reject_reason").isNotNull())
    if n_rejected_rows > 0:
        rejected.write.mode("overwrite").parquet(args.rejected)

    valid = flagged.filter(F.col("reject_reason").isNull()).drop("reject_reason")

    # --- R5 : doublons --------------------------------------------------------
    dedup = valid.dropDuplicates(DEDUP_KEYS).cache()
    n_dedup = dedup.count()
    n_dupes = (n_raw - n_rejected_rows) - n_dedup

    # --- R6 : sessions aberrantes (bots) ---------------------------------------
    session_sizes = dedup.groupBy("user_session").count().withColumnRenamed("count", "session_events")
    q50, q99, q999 = session_sizes.approxQuantile("session_events", [0.5, 0.99, 0.999], 0.001)
    max_events = session_sizes.agg(F.max("session_events")).first()[0]

    joined = dedup.join(session_sizes, "user_session")
    clean = joined.filter(F.col("session_events") <= args.max_session_events).drop("session_events")

    # --- Enrichissement ----------------------------------------------------------
    cat_parts = F.split(F.col("category_code"), r"\.")
    clean = (
        clean
        .withColumn("category_main", F.coalesce(cat_parts.getItem(0), F.lit("unknown")))
        .withColumn("category_sub", F.coalesce(cat_parts.getItem(1), F.lit("unknown")))
        .withColumn("brand", F.coalesce(F.col("brand"), F.lit("unknown")))
        .withColumn("hour", F.hour("event_time"))
        .withColumn("day_of_week", F.dayofweek("event_time"))  # 1 = dimanche
        .withColumn("is_weekend", F.dayofweek("event_time").isin(1, 7))
        .withColumn(
            "part_of_day",
            F.when(F.col("hour") < 6, "night")
            .when(F.col("hour") < 12, "morning")
            .when(F.col("hour") < 18, "afternoon")
            .otherwise("evening"),
        )
    )

    clean.write.mode("overwrite").partitionBy("event_date").parquet(args.output)

    # --- Vérification et rapport ---------------------------------------------------
    silver = spark.read.parquet(args.output)
    n_silver = silver.count()
    n_bots = n_dedup - n_silver
    elapsed = time.time() - t0

    rules = [(k, v) for k, v in sorted(rule_counts.items())]
    rules.append(("R5_doublons", n_dupes))
    rules.append((f"R6_sessions_aberrantes(>{args.max_session_events})", n_bots))

    print("\n" + "=" * 66)
    print(f"{'Règle':<42}{'Lignes rejetées':>16}{'%':>7}")
    print("-" * 66)
    for name, cnt in rules:
        print(f"{name:<42}{cnt:>16,}{100 * cnt / n_raw:>7.2f}")
    print("-" * 66)
    print(f"{'Bronze (entrée)':<42}{n_raw:>16,}")
    print(f"{'Silver (sortie)':<42}{n_silver:>16,}{100 * n_silver / n_raw:>7.2f}")
    print(f"Taille d'une session : médiane={q50:.0f}, p99={q99:.0f}, p99.9={q999:.0f}, max={max_events}")
    print(f"Temps total : {elapsed:.1f} s")
    print("=" * 66)

    silver.groupBy("event_type").count().orderBy(F.desc("count")).show()
    silver.groupBy("category_main").count().orderBy(F.desc("count")).show(10)

    report = {
        "bronze_rows": n_raw,
        "silver_rows": n_silver,
        "rejected_by_rule": dict(rules),
        "session_size": {"median": q50, "p99": q99, "p999": q999, "max": max_events},
        "max_session_events_threshold": args.max_session_events,
        "elapsed_seconds": round(elapsed, 1),
    }
    try:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Rapport écrit : {args.report}")
    except Exception as e:
        print(f"(rapport non écrit : {e})")

    spark.stop()


if __name__ == "__main__":
    main()
