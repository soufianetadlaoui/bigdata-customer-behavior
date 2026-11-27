"""
Référence pandas (une seule machine, un seul cœur) : même agrégation par client que Spark, même CSV.

À lancer sur Windows, depuis la racine du dépôt (.venv). Pour libérer de la RAM, arrêtez d'abord Docker :
    docker compose down
    wsl --shutdown
    python benchmarks\\bench_pandas.py

Le temps inclut la lecture du CSV (comme pour Spark) et le calcul. Les colonnes inutiles ne sont pas lues
(usecols), ce qui avantage pandas : la comparaison est volontairement équitable envers lui.
"""
import argparse
import csv
import datetime as dt
import os
import statistics
import sys
import time

import pandas as pd

HEADER = ["timestamp", "experiment", "variant", "query", "cores", "rows",
          "seconds_median", "seconds_min", "seconds_max", "reps", "extra"]
COLS = ["event_time", "event_type", "user_id", "user_session", "price"]
DTYPES = {"event_time": object, "event_type": "category", "user_id": "int64",
          "user_session": object, "price": "float64"}


def user_agg(df):
    """Même calcul que q_user dans benchmark.py : nombre d'événements, sessions, montant, dernier événement."""
    out = df.groupby("user_id").agg(
        n=("event_type", "size"),
        sessions=("user_session", "nunique"),
        last_ts=("event_time", "max"),
    )
    monetary = df[df["event_type"] == "purchase"].groupby("user_id")["price"].sum().rename("monetary")
    return out.join(monetary).fillna({"monetary": 0.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/oct_sample10.csv")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="docs/results/benchmark.csv")
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        sys.exit(f"CSV introuvable : {a.csv}")

    totals, reads, n, df_mb = [], [], 0, 0.0
    try:
        for i in range(a.reps):
            t0 = time.perf_counter()
            df = pd.read_csv(a.csv, usecols=COLS, dtype=DTYPES)
            t1 = time.perf_counter()
            res = user_agg(df)
            t2 = time.perf_counter()
            n, df_mb = len(df), df.memory_usage(deep=True).sum() / 1e6
            totals.append(t2 - t0)
            reads.append(t1 - t0)
            print(f"  essai {i + 1}/{a.reps} : lecture {t1 - t0:.1f} s, calcul {t2 - t1:.1f} s, "
                  f"total {t2 - t0:.1f} s ({len(res):,} clients, DataFrame {df_mb:.0f} Mo)", flush=True)
            del df, res
    except MemoryError:
        sys.exit("MemoryError : mémoire insuffisante pour pandas sur ce fichier (résultat à documenter tel quel).")

    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), "pandas", "pandas_csv", "user_agg", 1, n,
                    round(statistics.median(totals), 3), round(min(totals), 3), round(max(totals), 3), len(totals),
                    f"read_s={statistics.median(reads):.1f};df_mb={df_mb:.0f}"])
    print(f"\nMédiane : {statistics.median(totals):.1f} s sur {n:,} lignes -> {a.out}")


if __name__ == "__main__":
    main()
