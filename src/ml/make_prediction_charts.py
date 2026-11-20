"""
Graphiques de la prédiction d'achat (sur Windows, dans le .venv, depuis la racine du dépôt) :

    python src\\ml\\make_prediction_charts.py
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


m = rd("purchase_model_metrics")
base = float(m["test_base_rate"].iloc[0])
labels = {"logistic_regression": "Régression\nlogistique", "random_forest": "Random\nForest", "gbt": "Gradient\nBoosting"}
names = [labels.get(x, x) for x in m["model"]]

# --- 1. Comparaison des modèles -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(names, m["auc_roc"], color="#2a6fdb")
axes[0].axhline(0.5, ls="--", color="gray")
axes[0].text(len(m) - 0.5, 0.52, "hasard", ha="right", color="dimgray",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
axes[0].set_ylim(0.4, 1.0)
axes[0].set_title("AUC-ROC (test)")
axes[1].bar(names, m["auc_pr"], color="#e0572a")
axes[1].axhline(base, ls="--", color="gray")
axes[1].text(len(m) - 0.5, base * 1.06, f"hasard = {base:.3f}", ha="right", color="dimgray",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
axes[1].set_title("AUC-PR (test)")
for ax in axes:
    for i, v in enumerate(ax.patches):
        ax.text(v.get_x() + v.get_width() / 2, v.get_height(), f"{v.get_height():.3f}", ha="center", va="bottom")
save(fig, "12_comparaison_modeles")

# --- 2. Lift par décile du modèle final -------------------------------------------------------------
d = rd("purchase_deciles")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(d["decile"], 100 * d["purchase_rate"], color="#2a6fdb")
axes[0].axhline(100 * base, ls="--", color="gray", label=f"Taux moyen {100 * base:.1f} %")
axes[0].set_xlabel("Décile de score (1 = scores les plus élevés)")
axes[0].set_ylabel("Taux d'achat réel (%)")
axes[0].set_title("Taux d'achat par décile de score")
axes[0].set_xticks(d["decile"])
axes[0].legend()
gain = np.cumsum(d["buyers"]) / d["buyers"].sum()
axes[1].plot(np.r_[0, d["decile"] * 10], np.r_[0, 100 * gain], marker="o", color="#e0572a", label="Modèle")
axes[1].plot([0, 100], [0, 100], ls="--", color="gray", label="Hasard")
axes[1].set_xlabel("% de clients contactés (par score décroissant)")
axes[1].set_ylabel("% des acheteurs captés")
axes[1].set_title("Courbe de gain cumulée")
axes[1].legend()
save(fig, "13_lift_deciles")

# --- 3. Importance des variables ----------------------------------------------------------------------
try:
    imp = rd("purchase_feature_importance").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"], color="#2a6fdb")
    ax.set_title("Variables les plus importantes (modèle final)")
    save(fig, "14_importance_variables")
except FileNotFoundError:
    print("(importance des variables absente)")

print("\n=== Métriques ===")
print(m.to_string(index=False))
top = d.iloc[0]
print(f"\nDécile 1 : {100 * top.purchase_rate:.1f} % d'achat, soit {top.purchase_rate / base:.1f}x le taux moyen ({100 * base:.1f} %)")
print(f"Les 20 % de clients les mieux scorés captent {100 * gain.iloc[1]:.0f} % des acheteurs")
