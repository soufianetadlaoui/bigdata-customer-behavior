"""
Graphiques des benchmarks (sur Windows, dans le .venv, depuis la racine du dépôt) :

    python benchmarks\\make_benchmark_charts.py

Lit docs/results/benchmark.csv (plusieurs lancements possibles : on garde le plus récent par mesure).
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "docs/results"
IMG = "docs/images"
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
COL = {"a": "#2a6fdb", "b": "#f2a33a", "c": "#e0572a", "d": "#6aa84f"}

b = pd.read_csv(os.path.join(RES, "benchmark.csv"))
b = b.sort_values("timestamp").drop_duplicates(["experiment", "variant", "query", "cores", "rows"], keep="last")


def sub(exp):
    return b[b["experiment"] == exp]


def num(extra, key):
    m = re.search(rf"{key}=(\d+)", str(extra))
    return int(m.group(1)) if m else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.png")


def value(d, variant, query, col="seconds_median"):
    v = d[(d["variant"] == variant) & (d["query"] == query)][col]
    return float(v.iloc[0]) if len(v) else np.nan


def grouped(ax, d, variants, queries, labels, qlabels):
    """Barres = médiane ; traits = plage min-max des mesures (la dispersion fait partie du résultat)."""
    x = np.arange(len(queries))
    w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        med = np.array([value(d, v, q) for q in queries])
        lo = np.array([value(d, v, q, "seconds_min") for q in queries])
        hi = np.array([value(d, v, q, "seconds_max") for q in queries])
        err = np.vstack([np.nan_to_num(med - lo), np.nan_to_num(hi - med)])
        bars = ax.bar(x + (i - (len(variants) - 1) / 2) * w, med, w, yerr=err, capsize=3,
                      label=labels[v], color=list(COL.values())[i], error_kw={"elinewidth": 1, "alpha": 0.7})
        for r, val, h in zip(bars, med, np.nan_to_num(hi, nan=0)):
            if not np.isnan(val):
                ax.text(r.get_x() + r.get_width() / 2, max(h, val), f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([qlabels[q] for q in queries])
    ax.set_ylabel("Temps (s) : médiane et plage min-max")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=8, frameon=False)


# --- 1. Formats -----------------------------------------------------------------------------------------------
d = sub("formats")
if len(d):
    variants = [v for v in ["csv_schema", "csv_inferschema", "parquet_partitioned"] if v in set(d["variant"])]
    sizes = {v: num(d[d["variant"] == v]["extra"].iloc[0], "size_mb") for v in variants}
    names = {"csv_schema": "CSV (schéma explicite)", "csv_inferschema": "CSV (schéma inféré)", "parquet_partitioned": "Parquet partitionné"}
    labels = {v: f"{names[v]} : {sizes[v]} Mo" for v in variants}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    grouped(ax, d, variants, ["light", "price", "user_agg"], labels,
            {"light": "Comptage par type\n(1 colonne)", "price": "Prix moyen des achats\n(2 colonnes)", "user_agg": "Agrégation par client\n(shuffle)"})
    ax.set_title(f"CSV vs Parquet ({int(d['rows'].iloc[0]):,} lignes, Spark local[{int(d['cores'].iloc[0])}])")
    save(fig, "16_benchmark_formats")
    print(d[["variant", "query", "seconds_median", "extra"]].to_string(index=False))
else:
    print("(expérience formats absente)")

# --- 2. Partitionnement ------------------------------------------------------------------------------------------
d = sub("partitioning")
if len(d):
    variants = [v for v in ["partitioned", "flat_random"] if v in set(d["variant"])]
    labels = {}
    for v in variants:
        e = d[d["variant"] == v]["extra"].iloc[0]
        labels[v] = {"partitioned": "Partitionné par jour", "flat_random": "Non partitionné"}[v] + f" ({num(e, 'files')} fichiers)"
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    grouped(ax, d, variants, ["day", "three_days", "full_scan"], labels,
            {"day": "1 jour", "three_days": "3 jours", "full_scan": "Tout le mois"})
    ax.set_title("Effet du partitionnement par date (Parquet)")
    save(fig, "17_benchmark_partitionnement")
else:
    print("(expérience partitioning absente)")

# --- 3. Scalabilité selon le volume ----------------------------------------------------------------------------------------
d = sub("datasize")
if len(d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for q, lab, c in [("user_agg", "Agrégation par client", COL["a"]), ("funnel", "Entonnoir par session", COL["c"])]:
        s = d[d["query"] == q].sort_values("rows")
        if s.empty:
            continue
        axes[0].errorbar(s["rows"] / 1e6, s["seconds_median"], marker="o", color=c, label=lab, capsize=3,
                         yerr=[s["seconds_median"] - s["seconds_min"], s["seconds_max"] - s["seconds_median"]])
        r0, t0 = s["rows"].iloc[0], s["seconds_median"].iloc[0]
        axes[0].plot(s["rows"] / 1e6, t0 * s["rows"] / r0, ls="--", color=c, alpha=0.4)
        axes[1].plot(s["rows"] / 1e6, s["rows"] / s["seconds_median"] / 1e6, marker="o", color=c, label=lab)
    axes[0].set_xlabel("Millions de lignes")
    axes[0].set_ylabel("Temps (s)")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("Temps vs volume (pointillés : croissance linéaire de référence)")
    axes[0].legend()
    axes[1].set_xlabel("Millions de lignes")
    axes[1].set_ylabel("Débit (millions de lignes / s)")
    axes[1].set_title("Débit de traitement")
    save(fig, "18_benchmark_volume")
    print(d.sort_values(["query", "rows"])[["variant", "query", "rows", "seconds_median"]].to_string(index=False))
else:
    print("(expérience datasize absente)")

# --- 4. Accélération selon le nombre de cœurs ------------------------------------------------------------------------------------
d = sub("cores")
if len(d) and 1 in set(d["cores"]):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmax = int(d["cores"].max())
    ax.plot([1, cmax], [1, cmax], ls="--", color="gray", label="Accélération idéale")
    print("\n=== Accélération (cœurs) ===")
    for q, lab, c in [("user_agg", "Agrégation par client", COL["a"]), ("funnel", "Entonnoir par session", COL["c"])]:
        s = d[d["query"] == q].sort_values("cores")
        if s.empty:
            continue
        t1 = float(s[s["cores"] == 1]["seconds_median"].iloc[0])
        sp = t1 / s["seconds_median"]
        ax.plot(s["cores"], sp, marker="o", color=c, label=lab)
        for cores, x, t in zip(s["cores"], sp, s["seconds_median"]):
            print(f"{q:<10} {int(cores)} cœur(s) : {t:7.1f} s  accélération x{x:.2f}  efficacité {100 * x / cores:.0f} %")
    ax.set_xticks(sorted(set(d["cores"])))
    ax.set_xlabel("Cœurs (Spark local[N])")
    ax.set_ylabel("Accélération par rapport à 1 cœur")
    ax.set_title("Scalabilité : effet du nombre de cœurs")
    ax.legend()
    save(fig, "19_benchmark_coeurs")
else:
    print("(expérience cores absente ou sans mesure à 1 cœur)")

# --- 5. pandas vs Spark -----------------------------------------------------------------------------------------------------------------------
fm, pdb = sub("formats"), sub("pandas")
items = []
if len(pdb):
    r = pdb.iloc[-1]
    items.append(("pandas\n(1 cœur, CSV)", float(r["seconds_median"]), float(r["seconds_min"]), float(r["seconds_max"]), COL["b"]))
if len(fm):
    for v, lab, c in [("csv_schema", "Spark local[4]\nCSV", COL["a"]), ("parquet_partitioned", "Spark local[4]\nParquet", COL["d"])]:
        t = value(fm, v, "user_agg")
        if not np.isnan(t):
            items.append((lab, t, value(fm, v, "user_agg", "seconds_min"), value(fm, v, "user_agg", "seconds_max"), c))
if len(items) >= 2:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    med = np.array([i[1] for i in items])
    err = np.vstack([med - np.array([i[2] for i in items]), np.array([i[3] for i in items]) - med])
    bars = ax.bar([i[0] for i in items], med, yerr=err, capsize=4, color=[i[4] for i in items],
                  error_kw={"elinewidth": 1, "alpha": 0.7})
    for r, i in zip(bars, items):
        ax.text(r.get_x() + r.get_width() / 2, i[3], f"{i[1]:.1f} s", ha="center", va="bottom")
    ax.set_ylabel("Temps (s) : médiane et plage min-max")
    ax.set_title("Agrégation par client : pandas vs Spark (même fichier, même résultat)")
    save(fig, "20_benchmark_pandas_spark")
else:
    print("(comparaison pandas vs Spark incomplète)")

# --- 6. Point à grande échelle : CSV complet -------------------------------------------------------------------------------------------
full, sample = sub("fullcsv"), sub("formats")
if len(full):
    print("\n=== CSV complet (mode local) ===")
    print(full[["query", "rows", "seconds_median", "extra"]].to_string(index=False))
    s = sample[(sample["variant"] == "csv_schema") & (sample["query"] == "light")]
    f = full[full["query"] == "light"]
    if len(s) and len(f):
        r_s, t_s = float(s["rows"].iloc[0]), float(s["seconds_median"].iloc[0])
        r_f, t_f = float(f["rows"].iloc[0]), float(f["seconds_median"].iloc[0])
        print(f"\nVolume x{r_f / r_s:.1f}  ->  temps x{t_f / t_s:.1f} (comptage par type sur CSV)")
