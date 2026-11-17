"""
Graphiques de la segmentation (à lancer sur Windows, dans le .venv, depuis la racine du dépôt) :

    python src\\ml\\make_segment_charts.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "docs/results"
IMG = "docs/images"
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def rd(name):
    return pd.read_csv(os.path.join(RES, f"{name}.csv"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, f"{name}.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {name}.png")


prof = rd("segment_profiles")
sizes = rd("segment_sizes")
ks = rd("k_selection")
sample = rd("segment_sample")

segments = list(prof["segment"])
palette = plt.get_cmap("tab10")
colors = {s: palette(i) for i, s in enumerate(segments)}
colors["Panier sans achat"] = "#b0b0b0"
colors["Visiteurs"] = "#d9d9d9"

# --- 1. Choix de k ---------------------------------------------------------------------------
best = int(ks.loc[ks["chosen"] == 1, "k"].iloc[0])
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(ks["k"], ks["wssse"], marker="o", color="#2a6fdb")
axes[0].set_title("Méthode du coude (WSSSE)")
axes[0].set_xlabel("Nombre de clusters k")
axes[1].plot(ks["k"], ks["silhouette"], marker="o", color="#e0572a")
axes[1].set_title("Coefficient de silhouette")
axes[1].set_xlabel("Nombre de clusters k")
for ax in axes:
    ax.axvline(best, ls="--", color="gray")
    ax.text(best, ax.get_ylim()[1], f" k = {best}", va="top", color="gray")
save(fig, "08_choix_k")

# --- 2. Taille et chiffre d'affaires ---------------------------------------------------------------
s = sizes.sort_values("users")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].barh(s["segment"], 100 * s["user_share"], color=[colors.get(x, "gray") for x in s["segment"]])
axes[0].set_title("Part des clients (%)")
r = sizes[sizes["revenue_share"] > 0].sort_values("revenue_share")
axes[1].barh(r["segment"], 100 * r["revenue_share"], color=[colors.get(x, "gray") for x in r["segment"]])
axes[1].set_title("Part du chiffre d'affaires (%)")
save(fig, "09_taille_segments")

# --- 3. Profil des segments (écart à la moyenne) ----------------------------------------------------
cols = ["avg_recency_days", "avg_frequency", "avg_monetary", "avg_sessions", "avg_categories"]
labels = ["Récence (jours,\nrouge = ancien)", "Fréquence", "Montant", "Sessions", "Catégories"]
vals = prof[cols].values
z = (vals - vals.mean(axis=0)) / np.where(vals.std(axis=0) == 0, 1, vals.std(axis=0))
fig, ax = plt.subplots(figsize=(8, 1.2 + 0.8 * len(prof)))
im = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
for i in range(len(prof)):
    for j in range(len(cols)):
        ax.text(j, i, f"{vals[i, j]:,.1f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(labels)
ax.set_yticks(range(len(prof)))
ax.set_yticklabels([f"{a} (n={n:,})" for a, n in zip(prof["segment"], prof["users"])])
ax.set_title("Profil moyen des segments (couleur = écart relatif entre segments)")
ax.grid(False)
fig.colorbar(im, ax=ax, label="Écart standardisé")
save(fig, "10_profil_segments")

# --- 4. Nuage de points : récence x montant -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for seg in segments:
    d = sample[sample["segment"] == seg]
    ax.scatter(d["recency_days"], d["monetary"], s=10, alpha=0.5, color=colors[seg], label=seg)
ax.set_yscale("log")
ax.set_xlabel("Récence (jours depuis le dernier achat)")
ax.set_ylabel("Montant total dépensé (échelle log)")
ax.set_title("Acheteurs : récence vs montant, par segment")
ax.legend(markerscale=2)
save(fig, "11_nuage_segments")

print("\n=== Profils ===")
print(prof.to_string(index=False))
print("\n=== Tailles ===")
print(sizes.to_string(index=False))
