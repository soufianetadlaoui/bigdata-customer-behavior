"""
Génère les graphiques du rapport à partir des CSV produits par eda_funnel.py.
À lancer sur Windows, dans le .venv (pandas + matplotlib), depuis la racine du dépôt :

    python src\\processing\\make_charts.py
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


# --- KPI --------------------------------------------------------------------------
k = rd("kpis").iloc[0]
print("\n=== KPI ===")
print(f"Événements : {int(k.events):,} | Clients : {int(k.users):,} | Sessions : {int(k.sessions):,}")
print(f"Produits : {int(k.products):,} | Marques : {int(k.brands):,}")
print(f"Achats : {int(k.purchases):,} | Acheteurs : {int(k.buyers):,} | CA : {k.revenue:,.0f}")
print(f"Taux de conversion client : {100 * k.buyers / k.users:.2f} %")
print(f"Panier moyen (par achat)  : {k.revenue / k.purchases:,.2f}\n")

# --- 1. Évolution quotidienne ------------------------------------------------------------
d = rd("daily")
d["event_date"] = pd.to_datetime(d["event_date"])
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(d["event_date"], d["views"], color="#2a6fdb", label="Vues")
ax.set_ylabel("Vues par jour", color="#2a6fdb")
ax2 = ax.twinx()
ax2.plot(d["event_date"], d["purchases"], color="#e0572a", label="Achats")
ax2.set_ylabel("Achats par jour", color="#e0572a")
ax2.grid(False)
ax.set_title("Évolution quotidienne : vues et achats")
fig.autofmt_xdate()
save(fig, "01_evolution_quotidienne")

# --- 2. Heatmap heure x jour --------------------------------------------------------------
h = rd("hourly")
pivot = h.pivot(index="day_of_week", columns="hour", values="events").fillna(0)
labels = {1: "Dim", 2: "Lun", 3: "Mar", 4: "Mer", 5: "Jeu", 6: "Ven", 7: "Sam"}
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([labels[i] for i in pivot.index])
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Heure (UTC)")
ax.set_title("Activité par jour de la semaine et heure")
ax.grid(False)
fig.colorbar(im, ax=ax, label="Événements")
save(fig, "02_activite_heure_jour")

# --- 3. Entonnoir de conversion ------------------------------------------------------------
f = rd("funnel").iloc[0]
stages = ["Sessions", "Sessions avec panier", "Sessions avec achat"]
vals = [f.sessions, f.with_cart, f.with_purchase]
fig, ax = plt.subplots(figsize=(8, 3.8))
bars = ax.barh(stages[::-1], vals[::-1], color=["#e0572a", "#f2a33a", "#2a6fdb"])
for b, v in zip(bars, vals[::-1]):
    ax.text(v, b.get_y() + b.get_height() / 2, f"  {int(v):,} ({100 * v / f.sessions:.1f} %)", va="center")
ax.set_xlim(0, f.sessions * 1.35)
ax.set_title(f"Entonnoir de conversion (abandon de panier : {100 * f.cart_abandon_rate:.1f} %)")
ax.grid(False)
save(fig, "03_entonnoir")

# --- 4. Conversion par catégorie --------------------------------------------------------------
c = rd("funnel_category")
c = c[(c["category_main"] != "unknown") & (c["sessions"] >= 1000)].sort_values("purchase_rate")
c = c.tail(12)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(c["category_main"], 100 * c["purchase_rate"], color="#2a6fdb")
ax.set_xlabel("Sessions avec achat (%)")
ax.set_title("Taux de conversion par catégorie (hors catégorie inconnue)")
save(fig, "04_conversion_categorie")

# --- 5. Conversion par tranche de prix ---------------------------------------------------------
p = rd("funnel_price_band").sort_values("price_band")
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(p))
ax.bar(x - 0.2, 100 * p["cart_rate"], width=0.4, label="Sessions avec panier", color="#f2a33a")
ax.bar(x + 0.2, 100 * p["purchase_rate"], width=0.4, label="Sessions avec achat", color="#2a6fdb")
ax.set_xticks(x)
ax.set_xticklabels([s.split("_", 1)[1] for s in p["price_band"]])
ax.set_xlabel("Tranche de prix")
ax.set_ylabel("% des sessions")
ax.set_title("Comportement selon la tranche de prix")
ax.legend()
save(fig, "05_conversion_prix")

# --- 6. Rétention par cohorte --------------------------------------------------------------------
co = rd("cohorts")
mat = co.pivot(index="cohort_week", columns="week", values="users")
ret = mat.div(np.diag(mat.reindex(columns=mat.index).values), axis=0) * 100
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(ret.values, cmap="Blues", vmin=0, vmax=100)
for i in range(ret.shape[0]):
    for j in range(ret.shape[1]):
        v = ret.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > 55 else "black")
ax.set_xticks(range(ret.shape[1]))
ax.set_xticklabels([f"S{int(w)}" for w in ret.columns])
ax.set_yticks(range(ret.shape[0]))
ax.set_yticklabels([f"Cohorte S{int(w)}" for w in ret.index])
ax.set_title("Rétention des clients par semaine de première visite (%)")
ax.grid(False)
fig.colorbar(im, ax=ax)
save(fig, "06_retention_cohortes")

# --- 7. Top marques ---------------------------------------------------------------------------------
b = rd("top_brands").sort_values("revenue").tail(12)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(b["brand"], b["revenue"], color="#2a6fdb")
ax.set_xlabel("Chiffre d'affaires")
ax.set_title("Top marques par chiffre d'affaires")
save(fig, "07_top_marques")

# --- Autres résultats à citer dans le rapport -----------------------------------------------------------
print("\n=== Temps entre panier et achat (minutes) ===")
try:
    print(rd("time_to_purchase").to_string(index=False))
except FileNotFoundError:
    print("(fichier absent)")
print("\n=== Acheteurs selon le nombre d'achats ===")
print(rd("repeat_buyers").to_string(index=False))
