"""
Producteur d'événements : rejoue une tranche horaire du Silver sous forme de fichiers JSON.
1 fichier = 1 minute d'activité réelle, déposé toutes les N secondes (accéléré).

À lancer sur Windows, depuis la racine du dépôt (.venv avec pandas + pyarrow) :
    python src\\streaming\\stream_producer.py --clean
    python src\\streaming\\stream_producer.py --date 2019-10-15 --start-hour 14 --minutes 90 --interval 2

Les fichiers sont écrits sous un nom temporaire puis renommés : Spark ne lit jamais un fichier incomplet.
"""
import argparse
import os
import shutil
import sys
import time

import pandas as pd

COLS = ["event_time", "event_type", "user_id", "user_session",
        "product_id", "category_code", "brand", "price"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="data/silver/events")
    ap.add_argument("--out", default="data/stream/incoming")
    ap.add_argument("--date", default="2019-10-15", help="Jour à rejouer (AAAA-MM-JJ)")
    ap.add_argument("--start-hour", type=int, default=14, help="Heure de début (UTC)")
    ap.add_argument("--minutes", type=int, default=90, help="Durée à rejouer, en minutes d'activité réelle")
    ap.add_argument("--interval", type=float, default=2.0, help="Secondes entre deux fichiers")
    ap.add_argument("--clean", action="store_true", help="Vide le dossier d'entrée avant de commencer")
    args = ap.parse_args()

    part = os.path.join(args.silver, f"event_date={args.date}")
    if not os.path.isdir(part):
        sys.exit(f"Partition introuvable : {part}")

    if args.clean:
        shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(part, columns=COLS)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True).dt.tz_localize(None)
    start = pd.Timestamp(f"{args.date} {args.start_hour:02d}:00:00")
    end = start + pd.Timedelta(minutes=args.minutes)
    df = df[(df["event_time"] >= start) & (df["event_time"] < end)].sort_values("event_time")
    if df.empty:
        sys.exit("Aucun événement dans cette tranche horaire : changez --date ou --start-hour.")

    df["minute"] = df["event_time"].dt.floor("min")
    df["event_time"] = df["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    n_files = df["minute"].nunique()
    print(f"{len(df):,} événements du {start} au {end} (UTC), {n_files} fichiers, "
          f"un toutes les {args.interval} s (environ {int(n_files * args.interval)} s au total).")

    for i, (minute, chunk) in enumerate(df.groupby("minute", sort=True)):
        name = f"batch_{i:05d}_{minute:%H%M}.json"
        tmp = os.path.join(args.out, f".tmp_{name}")  # Spark ignore les fichiers commençant par un point
        chunk[COLS].to_json(tmp, orient="records", lines=True)
        os.replace(tmp, os.path.join(args.out, name))
        types = chunk["event_type"].value_counts()
        print(f"[{i + 1:>3}/{n_files}] {minute:%H:%M}  {len(chunk):>5} événements  "
              f"(vues {types.get('view', 0)}, paniers {types.get('cart', 0)}, achats {types.get('purchase', 0)})",
              flush=True)
        time.sleep(args.interval)

    print("\nRejeu terminé. Laissez Spark tourner encore une minute : les derniers paniers sont signalés "
          "quand le filigrane (watermark) avance.")


if __name__ == "__main__":
    main()
