"""
Étape 8a : benchmarks de performance et de scalabilité (Spark en mode local).

Usage (dans le conteneur) :
  docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g \
      /opt/project/benchmarks/benchmark.py --experiment formats

Expériences :
  formats       CSV (schéma explicite / schéma inféré) vs Parquet partitionné, 3 requêtes
  partitioning  Parquet partitionné par jour vs copie non partitionnée (filtre 1 jour / 3 jours / aucun)
  datasize      Même traitement sur 10 %, 25 %, 50 %, 100 % des clients du Silver
  cores         Même traitement sur 100 % du Silver : à lancer avec --master local[1], local[2], local[4]
  fullcsv       Lecture du CSV complet d'octobre (~42 M de lignes), une seule mesure par requête (long)
  all           formats + partitioning + datasize + cores (avec le --master courant)

Méthode : une exécution de chauffe (JVM, cache disque du système), puis N mesures ; médiane retenue.
Les requêtes sont exécutées jusqu'au bout avec un puits "noop" (aucune écriture sur disque).
Les temps incluent la lecture des fichiers, pas seulement le calcul. Le cache disque du système
d'exploitation reste actif : les mesures décrivent un régime "données déjà lues une fois".
Résultats ajoutés à docs/results/benchmark.csv (ne pas lancer deux expériences en même temps).
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import shutil
import statistics
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

CSV_SCHEMA = StructType([
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
HEADER = ["timestamp", "experiment", "variant", "query", "cores", "rows",
          "seconds_median", "seconds_min", "seconds_max", "reps", "extra"]


# --- Utilitaires ----------------------------------------------------------------------------------
def dir_size_mb(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6


def count_parquet_files(path):
    return sum(1 for _, _, fs in os.walk(path) for f in fs if f.endswith(".parquet"))


def append_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerows(rows)


def run(df):
    """Exécute le plan complet sans rien écrire (puits 'noop')."""
    df.write.format("noop").mode("overwrite").save()


def timed(fn, reps, warmup=True):
    if warmup:
        fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


# --- Requêtes ----------------------------------------------------------------------------------------
def q_light(df):  # une colonne : favorise le format colonnaire
    return df.groupBy("event_type").count()


def q_price(df):  # deux colonnes avec filtre
    return df.filter(F.col("event_type") == "purchase").agg(F.avg("price"), F.sum("price"))


def q_user(df):  # agrégation par client : shuffle important
    et = F.col("event_type")
    return df.groupBy("user_id").agg(
        F.count("*").alias("n"),
        F.countDistinct("user_session").alias("sessions"),
        F.sum(F.when(et == "purchase", F.col("price")).otherwise(0.0)).alias("monetary"),
        F.max("event_time").alias("last_ts"),
    )


def q_funnel(df):  # agrégation par session puis globale
    et = F.col("event_type")
    s = df.groupBy("user_session").agg(
        F.max((et == "view").cast("int")).alias("v"),
        F.max((et == "cart").cast("int")).alias("c"),
        F.max((et == "purchase").cast("int")).alias("p"),
    )
    return s.agg(F.count("*"), F.sum("v"), F.sum("c"), F.sum("p"))


QUERIES = {"light": q_light, "price": q_price, "user_agg": q_user, "funnel": q_funnel}


# --- Expériences ----------------------------------------------------------------------------------------
def exp_formats(spark, a, rec):
    cores = spark.sparkContext.defaultParallelism
    n = spark.read.parquet(a.bronze).count()
    sources = []
    if os.path.exists(a.csv):
        mb = os.path.getsize(a.csv) / 1e6
        sources.append(("csv_schema", lambda: spark.read.csv(a.csv, header=True, schema=CSV_SCHEMA),
                        mb, ["light", "price", "user_agg"]))
        sources.append(("csv_inferschema", lambda: spark.read.csv(a.csv, header=True, inferSchema=True),
                        mb, ["light"]))
    else:
        print(f"(CSV absent : {a.csv}, variantes CSV ignorées)")
    sources.append(("parquet_partitioned", lambda: spark.read.parquet(a.bronze),
                    dir_size_mb(a.bronze), ["light", "price", "user_agg"]))
    for name, make, mb, qs in sources:
        for qn in qs:
            q = QUERIES[qn]
            rec("formats", name, qn, cores, n, timed(lambda: run(q(make())), a.reps), f"size_mb={mb:.0f}")


def exp_partitioning(spark, a, rec):
    cores = spark.sparkContext.defaultParallelism
    flat = os.path.join(a.scratch, "_bench_flat")
    shutil.rmtree(flat, ignore_errors=True)
    src = spark.read.parquet(a.bronze)
    n = src.count()
    src.repartition(31).write.mode("overwrite").parquet(flat)  # même volume, répartition aléatoire, sans partitions
    d = lambda s: F.lit(s).cast("date")
    qs = {
        "day": lambda df: df.filter(F.col("event_date") == d("2019-10-15")).groupBy("event_type").count(),
        "three_days": lambda df: df.filter(F.col("event_date").between(d("2019-10-14"), d("2019-10-16")))
                                   .groupBy("event_type").count(),
        "full_scan": lambda df: df.groupBy("event_type").count(),
    }
    variants = [("partitioned", a.bronze), ("flat_random", flat)]
    for name, path in variants:
        extra = f"size_mb={dir_size_mb(path):.0f};files={count_parquet_files(path)}"
        for qn, q in qs.items():
            rec("partitioning", name, qn, cores, n,
                timed(lambda: run(q(spark.read.parquet(path))), a.reps), extra)
    shutil.rmtree(flat, ignore_errors=True)


def silver_fraction(spark, a, pct):
    df = spark.read.parquet(a.silver)
    # fraction de CLIENTS (et non de lignes) : on garde leurs historiques complets
    return df.filter(F.pmod(F.hash("user_id"), F.lit(100)) < pct)


def exp_datasize(spark, a, rec):
    cores = spark.sparkContext.defaultParallelism
    for pct in (10, 25, 50, 100):
        n = silver_fraction(spark, a, pct).count()
        for qn in ("user_agg", "funnel"):
            q = QUERIES[qn]
            rec("datasize", f"{pct}pct_clients", qn, cores, n,
                timed(lambda: run(q(silver_fraction(spark, a, pct))), a.reps))


def exp_cores(spark, a, rec):
    cores = spark.sparkContext.defaultParallelism
    n = silver_fraction(spark, a, 100).count()
    for qn in ("user_agg", "funnel"):
        q = QUERIES[qn]
        rec("cores", "100pct_clients", qn, cores, n, timed(lambda: run(q(silver_fraction(spark, a, 100))), a.reps))


def exp_fullcsv(spark, a, rec):
    if not os.path.exists(a.full_csv):
        return print(f"CSV complet absent : {a.full_csv}")
    cores = spark.sparkContext.defaultParallelism
    mb = os.path.getsize(a.full_csv) / 1e6
    make = lambda: spark.read.csv(a.full_csv, header=True, schema=CSV_SCHEMA)
    t = time.perf_counter()
    n = make().count()
    rec("fullcsv", "csv_schema", "count", cores, n, [time.perf_counter() - t], f"size_mb={mb:.0f}")
    for qn in ("light", "price"):
        q = QUERIES[qn]
        rec("fullcsv", "csv_schema", qn, cores, n, timed(lambda: run(q(make())), 1, warmup=False), f"size_mb={mb:.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   choices=["formats", "partitioning", "datasize", "cores", "fullcsv", "all"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--csv", default="/opt/project/data/raw/oct_sample10.csv")
    p.add_argument("--full-csv", default="/opt/project/data/raw/2019-Oct.csv")
    p.add_argument("--bronze", default="/opt/project/data/bronze/events")
    p.add_argument("--silver", default="/opt/project/data/silver/events")
    p.add_argument("--scratch", default="/opt/project/data/gold", help="Dossier (ignoré par Git) pour la copie non partitionnée")
    p.add_argument("--out", default="/opt/project/docs/results/benchmark.csv")
    a = p.parse_args()

    spark = (
        SparkSession.builder.appName("benchmark")
        .config("spark.sql.shuffle.partitions", "16")  # fixé pour toutes les mesures
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext
    ts = dt.datetime.now().isoformat(timespec="seconds")
    rows = []

    def rec(exp, variant, query, cores, n, times, extra=""):
        med = statistics.median(times)
        rows.append([ts, exp, variant, query, cores, n, round(med, 3), round(min(times), 3),
                     round(max(times), 3), len(times), extra])
        print(f"{exp:<13}{variant:<22}{query:<11}cœurs={cores:<3}lignes={n:>11,}  médiane={med:8.2f} s "
              f"[{min(times):.2f} ; {max(times):.2f}]  {extra}", flush=True)

    exps = {"formats": exp_formats, "partitioning": exp_partitioning, "datasize": exp_datasize,
            "cores": exp_cores, "fullcsv": exp_fullcsv}
    chosen = ["formats", "partitioning", "datasize", "cores"] if a.experiment == "all" else [a.experiment]
    for e in chosen:
        print(f"\n=== {e} (master={sc.master}, parallélisme={sc.defaultParallelism}) ===", flush=True)
        exps[e](spark, a, rec)
        append_rows(a.out, rows)  # écriture après chaque expérience : rien n'est perdu en cas d'arrêt
        rows.clear()

    env = {"date": ts, "spark": spark.version, "master": sc.master, "parallelisme": sc.defaultParallelism,
           "cpu_visibles_conteneur": os.cpu_count(), "driver_memory": sc.getConf().get("spark.driver.memory", "n/a"),
           "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"), "python": platform.python_version()}
    with open(a.out.replace(".csv", "_env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    print(f"\nRésultats ajoutés à {a.out}")
    spark.stop()


if __name__ == "__main__":
    main()
