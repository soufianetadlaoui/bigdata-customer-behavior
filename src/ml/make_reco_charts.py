"""
Graphique de la recommandation (sur Windows, dans le .venv, depuis la racine du dépôt) :

    python src\\ml\\make_reco_charts.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "docs/results"
IMG = "docs/images"
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

m = pd.read_csv(os.path.join(RES, "recommender_metrics.csv"))
labels = {
    "popularite_achats": "Produits les plus\nachetés",
    "produits_recents": "Produits récemment\nconsultés",
    "als": "ALS",
}
metrics = [("precision_at_10", "Precision@10"), ("recall_at_10", "Recall@10"), ("hit_rate_at_10", "HitRate@10")]
colors = ["#b0b0b0", "#f2a33a", "#2a6fdb"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (col, title) in zip(axes, metrics):
    bars = ax.bar([labels.get(x, x) for x in m["model"]], 100 * m[col], color=colors[: len(m)])
    ax.set_title(title + " (%)")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}", ha="center", va="bottom")
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle(f"Recommandation : produits achetés du 22 au 31 octobre ({int(m['users_evaluated'].iloc[0]):,} clients)")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "15_comparaison_recommandation.png"), dpi=150)
plt.close(fig)
print("  -> 15_comparaison_recommandation.png\n")
print(m.to_string(index=False))

best = m.loc[m["hit_rate_at_10"].idxmax(), "model"]
als = m[m["model"] == "als"].iloc[0]
rec = m[m["model"] == "produits_recents"].iloc[0]
print(f"\nMeilleure méthode (HitRate@10) : {best}")
print(f"ALS vs reciblage : HitRate {100 * als.hit_rate_at_10:.1f} % contre {100 * rec.hit_rate_at_10:.1f} %")
