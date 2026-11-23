"""
Tableau de bord : analyse du comportement client à grande échelle.

Lancement (depuis la racine du dépôt, dans le .venv) :
    pip install streamlit plotly pandas pyarrow
    streamlit run dashboard/app.py

Sources de données :
  - docs/results/*.csv    : agrégats produits par les scripts Spark (versionnés dans Git)
  - data/gold/*           : tables client par client (non versionnées) pour le ciblage et la fiche client
Les pages qui dépendent de data/gold affichent un message si ces tables sont absentes.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
GOLD = ROOT / "data" / "gold"
DAYS = {1: "Dim", 2: "Lun", 3: "Mar", 4: "Mer", 5: "Jeu", 6: "Ven", 7: "Sam"}

st.set_page_config(page_title="Comportement client - Big Data", layout="wide")


@st.cache_data(show_spinner=False)
def csv(name):
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner="Chargement des données...")
def gold(name):
    p = GOLD / name
    return pd.read_parquet(p) if p.exists() else None


def show(fig, where=st):
    """Affiche un graphique Plotly à la largeur du conteneur (compatible anciennes et nouvelles versions)."""
    try:
        where.plotly_chart(fig, use_container_width=True)
    except TypeError:
        where.plotly_chart(fig, width="stretch")


def table(df, where=st):
    try:
        where.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        where.dataframe(df, width="stretch", hide_index=True)


def read_json_dir(folder):
    """Lit les fichiers JSON (une ligne = un enregistrement) écrits par Spark dans un dossier."""
    if not folder.exists():
        return None
    files = [f for f in sorted(folder.glob("part-*.json")) if f.stat().st_size > 0]
    return pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True) if files else None


def missing(what):
    st.warning(f"Données absentes : {what}. Exécutez l'étape correspondante du pipeline.")


# --------------------------------------------------------------------------------------------
def page_overview():
    st.header("Vue d'ensemble")
    k = csv("kpis")
    if k is None:
        return missing("docs/results/kpis.csv")
    k = k.iloc[0]
    c = st.columns(5)
    c[0].metric("Clients", f"{int(k['users']):,}")
    c[1].metric("Sessions", f"{int(k['sessions']):,}")
    c[2].metric("Acheteurs", f"{int(k['buyers']):,}",
                f"{100 * k['buyers'] / k['users']:.1f} % des clients", delta_color="off")
    c[3].metric("Chiffre d'affaires", f"{k['revenue']:,.0f}")
    c[4].metric("CA par acheteur", f"{k['revenue'] / k['buyers']:,.0f}")

    d = csv("daily")
    if d is not None:
        d["event_date"] = pd.to_datetime(d["event_date"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=d["event_date"], y=d["views"], name="Vues"), secondary_y=False)
        fig.add_trace(go.Scatter(x=d["event_date"], y=d["purchases"], name="Achats"), secondary_y=True)
        fig.update_yaxes(title_text="Vues", secondary_y=False)
        fig.update_yaxes(title_text="Achats", secondary_y=True)
        fig.update_layout(title="Activité quotidienne", height=380, legend_orientation="h")
        show(fig, st)

    h = csv("hourly")
    if h is not None:
        pv = h.pivot(index="day_of_week", columns="hour", values="events").fillna(0)
        pv.index = [DAYS[i] for i in pv.index]
        fig = px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd",
                        labels=dict(x="Heure (UTC)", y="", color="Événements"),
                        title="Activité par jour de la semaine et par heure")
        show(fig, st)


# --------------------------------------------------------------------------------------------
def page_funnel():
    st.header("Parcours client et conversion")
    f = csv("funnel_product")
    if f is not None:
        f = f.iloc[0]
        stages = pd.DataFrame({
            "Étape": ["Produits consultés", "Consultés puis mis au panier", "Mis au panier puis achetés"],
            "Couples client-produit": [f["viewed"], f["viewed_carted"], f["viewed_carted_purchased"]],
        })
        a, b = st.columns([2, 1])
        show(px.funnel(stages, x="Couples client-produit", y="Étape"), a)
        b.metric("Abandon de panier", f"{100 * f['cart_abandon_rate']:.1f} %")
        b.metric("Achats sans panier préalable", f"{100 * f['direct_purchase_share']:.1f} %")
        st.caption(
            "Entonnoir calculé par couple (client, produit) sur tout le mois. Plus de la moitié des produits "
            "achetés n'ont aucun événement « panier » préalable : l'entonnoir ne décrit donc que le chemin via le panier."
        )

    left, right = st.columns(2)
    cat = csv("funnel_category")
    if cat is not None:
        with_unknown = left.checkbox("Inclure la catégorie inconnue", value=False)
        cc = cat[cat["sessions"] >= 1000]
        if not with_unknown:
            cc = cc[cc["category_main"] != "unknown"]
        cc = cc.sort_values("purchase_rate")
        fig = px.bar(cc, x="purchase_rate", y="category_main", orientation="h",
                     labels={"purchase_rate": "Sessions avec achat", "category_main": ""},
                     title="Conversion par catégorie")
        fig.update_xaxes(tickformat=".1%")
        show(fig, left)

    price = csv("funnel_price_band")
    if price is not None:
        pp = price.sort_values("price_band").melt(
            id_vars="price_band", value_vars=["cart_rate", "purchase_rate"],
            var_name="indicateur", value_name="taux")
        pp["indicateur"] = pp["indicateur"].map({"cart_rate": "Panier", "purchase_rate": "Achat"})
        pp["price_band"] = pp["price_band"].str.split("_", n=1).str[1]
        fig = px.bar(pp, x="price_band", y="taux", color="indicateur", barmode="group",
                     labels={"price_band": "Tranche de prix", "taux": "Part des sessions"},
                     title="Comportement selon la tranche de prix")
        fig.update_yaxes(tickformat=".1%")
        show(fig, right)

    t = csv("time_to_purchase")
    rep = csv("repeat_buyers")
    c1, c2 = st.columns(2)
    if t is not None:
        q = t["quantile"].round(2)
        c1.subheader("Délai entre panier et achat")
        c1.metric("Médiane", f"{t.loc[q == 0.5, 'minutes'].iloc[0]:.1f} min")
        c1.metric("90 % des achats en moins de", f"{t.loc[q == 0.9, 'minutes'].iloc[0]:.1f} min")
    if rep is not None:
        fig = px.bar(rep, x="purchases", y="buyers",
                     labels={"purchases": "Articles achetés", "buyers": "Acheteurs"},
                     title="Acheteurs selon le nombre d'articles achetés")
        show(fig, c2)
        c2.caption("Compte des articles, pas des commandes : borne haute de la récurrence.")


# --------------------------------------------------------------------------------------------
def page_segments():
    st.header("Segmentation des clients (RFM + K-Means)")
    sizes, prof = csv("segment_sizes"), csv("segment_profiles")
    ks, sample = csv("k_selection"), csv("segment_sample")
    if sizes is None:
        return missing("docs/results/segment_sizes.csv")

    m = sizes.melt(id_vars="segment", value_vars=["user_share", "revenue_share"],
                   var_name="indicateur", value_name="part")
    m["indicateur"] = m["indicateur"].map({"user_share": "Part des clients", "revenue_share": "Part du CA"})
    fig = px.bar(m, x="segment", y="part", color="indicateur", barmode="group",
                 category_orders={"segment": list(sizes["segment"])},
                 labels={"segment": "", "part": ""}, title="Poids des segments : clients vs chiffre d'affaires")
    fig.update_yaxes(tickformat=".0%")
    show(fig, st)

    if prof is not None:
        st.subheader("Profil moyen des segments d'acheteurs")
        table(prof.drop(columns=["cluster"]).rename(columns={
            "segment": "Segment", "users": "Clients", "avg_recency_days": "Récence (j)",
            "avg_frequency": "Fréquence", "avg_monetary": "Montant", "avg_sessions": "Sessions",
            "avg_categories": "Catégories", "revenue": "CA"}))

    a, b = st.columns(2)
    if sample is not None:
        fig = px.scatter(sample, x="recency_days", y="monetary", color="segment", log_y=True, opacity=0.6,
                         labels={"recency_days": "Récence (jours)", "monetary": "Montant dépensé"},
                         title="Acheteurs : récence vs montant (échantillon)")
        show(fig, a)
    if ks is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=ks["k"], y=ks["wssse"], name="WSSSE (coude)", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Scatter(x=ks["k"], y=ks["silhouette"], name="Silhouette", mode="lines+markers"), secondary_y=True)
        fig.update_xaxes(title_text="Nombre de clusters k")
        fig.update_layout(title="Choix du nombre de segments", legend_orientation="h")
        show(fig, b)
        b.caption("La silhouette reste inférieure à 0,4 pour k de 3 à 8 : les comportements forment un continuum.")


# --------------------------------------------------------------------------------------------
def page_prediction():
    st.header("Prédiction de l'achat à 10 jours et ciblage")
    m, dec, imp = csv("purchase_model_metrics"), csv("purchase_deciles"), csv("purchase_feature_importance")
    if m is None:
        return missing("docs/results/purchase_model_metrics.csv")
    base = float(m["test_base_rate"].iloc[0])

    a, b = st.columns(2)
    mm = m.melt(id_vars="model", value_vars=["auc_roc", "auc_pr"], var_name="métrique", value_name="valeur")
    mm["métrique"] = mm["métrique"].map({"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR"})
    fig = px.bar(mm, x="model", y="valeur", color="métrique", barmode="group", text_auto=".3f",
                 labels={"model": "", "valeur": ""}, title="Comparaison des modèles (période de test)")
    show(fig, a)
    if dec is not None:
        fig = px.bar(dec, x="decile", y="purchase_rate",
                     labels={"decile": "Décile de score (1 = scores les plus élevés)", "purchase_rate": "Taux d'achat réel"},
                     title="Taux d'achat par décile de score")
        fig.add_hline(y=base, line_dash="dash", annotation_text=f"moyenne {base:.1%}")
        fig.update_yaxes(tickformat=".0%")
        show(fig, b)
        top20 = dec.sort_values("decile").head(2)["buyers"].sum() / dec["buyers"].sum()
        b.caption(f"Les 20 % de clients les mieux scorés captent {top20:.0%} des acheteurs.")
    st.caption("Les trois modèles obtiennent des performances proches ; le score sert à classer les clients, "
               "pas à estimer une probabilité absolue (le taux d'achat de base diffère entre entraînement et test).")
    if imp is not None:
        fig = px.bar(imp.head(12).iloc[::-1], x="importance", y="feature", orientation="h",
                     labels={"feature": "", "importance": "Importance"}, title="Variables les plus importantes")
        show(fig, st)

    st.subheader("Ciblage marketing")
    sc = gold("purchase_scores")
    if sc is None:
        return missing("data/gold/purchase_scores")
    segs = sorted(sc["segment"].dropna().unique())
    c1, c2 = st.columns([2, 1])
    chosen = c1.multiselect("Segments", segs, default=segs)
    pct = c2.slider("Part des clients les mieux scorés (%)", 1, 50, 10)
    sub = sc[sc["segment"].isin(chosen)]
    if sub.empty:
        return st.info("Sélectionnez au moins un segment.")
    top = sub.nlargest(max(int(len(sub) * pct / 100), 1), "purchase_prob")
    rate, avg = top["label"].mean(), sub["label"].mean()
    k1, k2, k3 = st.columns(3)
    k1.metric("Clients ciblés", f"{len(top):,}")
    k2.metric("Ont réellement acheté (test)", f"{rate:.1%}")
    k3.metric("Gain vs moyenne de la sélection", f"x{rate / avg:.1f}" if avg > 0 else "n/a")
    st.caption("« Ont réellement acheté » vient de la période de test (22-31 octobre) : c'est une validation, "
               "cette information ne serait pas connue en production.")
    st.download_button("Télécharger la liste (CSV)",
                       top[["user_id", "segment", "purchase_prob"]].to_csv(index=False).encode("utf-8"),
                       "ciblage_clients.csv", "text/csv")
    table(top[["user_id", "segment", "purchase_prob", "label"]].head(100)
          .rename(columns={"purchase_prob": "Score", "label": "Achat réel"}))


# --------------------------------------------------------------------------------------------
def page_reco():
    st.header("Recommandation de produits (ALS)")
    m = csv("recommender_metrics")
    if m is None:
        return missing("docs/results/recommender_metrics.csv")
    names = {"popularite_achats": "Produits les plus achetés", "produits_recents": "Produits récemment consultés", "als": "ALS"}
    mm = m.melt(id_vars="model", value_vars=["precision_at_10", "recall_at_10", "hit_rate_at_10"],
                var_name="métrique", value_name="valeur")
    mm["model"] = mm["model"].map(names)
    mm["métrique"] = mm["métrique"].map(
        {"precision_at_10": "Precision@10", "recall_at_10": "Recall@10", "hit_rate_at_10": "HitRate@10"})
    fig = px.bar(mm, x="métrique", y="valeur", color="model", barmode="group", text_auto=".1%",
                 labels={"valeur": "", "métrique": "", "model": "Méthode"},
                 title=f"Produits achetés du 22 au 31 octobre ({int(m['users_evaluated'].iloc[0]):,} clients)")
    fig.update_yaxes(tickformat=".0%")
    show(fig, st)
    st.caption("Le reciblage (produits récemment consultés) fait mieux qu'ALS sur ce jeu de données : "
               "beaucoup d'achats portent sur un produit déjà consulté. ALS apporte surtout de la découverte.")

    st.subheader("Fiche client")
    recs, cat = gold("recommendations"), gold("product_catalog")
    if recs is None or cat is None:
        return missing("data/gold/recommendations et data/gold/product_catalog")
    seg, sc = gold("customer_segments"), gold("purchase_scores")
    users = recs["user_id"].unique()

    def pick_random():
        st.session_state["uid"] = int(pd.Series(users).sample(1).iloc[0])

    st.session_state.setdefault("uid", int(users[0]))
    st.button("Client aléatoire", on_click=pick_random)
    uid = int(st.number_input("Identifiant client (parmi ceux ayant des recommandations)", step=1, key="uid"))

    r = recs[recs["user_id"] == uid].sort_values("rank")
    if r.empty:
        return st.info("Aucune recommandation pour ce client : le tableau n'en contient que pour un échantillon.")
    c = st.columns(4)
    if seg is not None:
        s = seg[seg["user_id"] == uid]
        if not s.empty:
            s = s.iloc[0]
            c[0].metric("Segment", str(s["segment"]))
            if s["segment_type"] == "buyer":
                c[1].metric("Montant dépensé", f"{s['monetary']:,.0f}")
                c[2].metric("Sessions d'achat", f"{int(s['frequency'])}")
    if sc is not None:
        p = sc[sc["user_id"] == uid]
        if not p.empty:
            prob = float(p["purchase_prob"].iloc[0])
            c[3].metric("Score d'achat (percentile)", f"{100 * (sc['purchase_prob'] < prob).mean():.0f} %")
    table(
        r.merge(cat, on="product_id", how="left")[["rank", "brand", "category_code", "avg_price", "score"]]
        .rename(columns={"rank": "Rang", "brand": "Marque", "category_code": "Catégorie",
                         "avg_price": "Prix moyen", "score": "Score ALS"}))


# --------------------------------------------------------------------------------------------
def page_streaming():
    st.header("Traitement en continu (démonstration)")
    st.caption("Événements rejoués par un producteur, consommés par Spark Structured Streaming. "
               "Paniers sans achat dans les 10 minutes : alerte de relance, enrichie du segment client.")
    st.button("Rafraîchir")
    base = ROOT / "data" / "stream" / "out"
    kpi = read_json_dir(base / "kpi_per_minute")
    if kpi is None:
        return missing("data/stream/out : lancez streaming_demo.py puis stream_producer.py")
    kpi["window_start"] = pd.to_datetime(kpi["window_start"])
    wide = (kpi.pivot_table(index="window_start", columns="event_type", values="events", aggfunc="sum")
            .fillna(0).reset_index())
    types = [t for t in ["view", "cart", "purchase"] if t in wide.columns]
    c = st.columns(4)
    c[0].metric("Minutes traitées", f"{len(wide):,}")
    c[1].metric("Événements", f"{int(wide[types].sum().sum()):,}")
    c[2].metric("Achats", f"{int(wide['purchase'].sum()) if 'purchase' in wide else 0:,}")
    c[3].metric("Chiffre d'affaires", f"{kpi['revenue'].sum():,.0f}")
    show(px.line(wide, x="window_start", y=types,
                 labels={"window_start": "Minute (temps des événements)", "value": "Événements", "variable": "Type"},
                 title="Événements par minute"))

    stats_p = base / "cart_stats.csv"
    if stats_p.exists():
        s = pd.read_csv(stats_p)
        conv, ab = int(s["converted"].sum()), int(s["abandoned"].sum())
        a, b, d = st.columns(3)
        a.metric("Paniers convertis en 10 min", f"{conv:,}")
        b.metric("Alertes d'abandon", f"{ab:,}")
        d.metric("Part sans achat", f"{100 * ab / max(conv + ab, 1):.1f} %")

    alerts = read_json_dir(base / "abandoned_carts")
    if alerts is not None:
        alerts["cart_time"] = pd.to_datetime(alerts["cart_time"])
        if "segment" not in alerts:
            alerts["segment"] = None
        alerts["segment"] = alerts["segment"].fillna("Inconnu")
        by = alerts.groupby("segment").size().reset_index(name="alertes").sort_values("alertes")
        left, right = st.columns([1, 2])
        show(px.bar(by, x="alertes", y="segment", orientation="h", title="Alertes par segment client",
                    labels={"segment": ""}), left)
        right.subheader("Dernières alertes")
        table(alerts.sort_values("cart_time", ascending=False).head(100)[
            ["cart_time", "user_id", "segment", "brand", "category_code", "price"]], right)
    else:
        st.info("Aucune alerte pour l'instant : elles apparaissent quand le filigrane dépasse l'heure du panier + 10 minutes.")


def page_data():
    st.header("Données et architecture")
    qp = ROOT / "docs" / "quality_report.json"
    if qp.exists():
        q = json.loads(qp.read_text(encoding="utf-8"))
        c = st.columns(3)
        c[0].metric("Lignes Bronze", f"{q['bronze_rows']:,}")
        c[1].metric("Lignes Silver", f"{q['silver_rows']:,}")
        c[2].metric("Conservées", f"{100 * q['silver_rows'] / q['bronze_rows']:.2f} %")
        rej = pd.DataFrame({"Règle": list(q["rejected_by_rule"]), "Lignes rejetées": list(q["rejected_by_rule"].values())})
        show(px.bar(rej, x="Règle", y="Lignes rejetées", title="Lignes rejetées par règle de qualité"))
        s = q.get("session_size", {})
        if s:
            st.caption(f"Taille d'une session : médiane {s.get('median')}, p99 {s.get('p99')}, "
                       f"maximum {s.get('max')} événements : aucune session ne dépasse le seuil de détection de bots.")
    else:
        missing("docs/quality_report.json")
    st.subheader("Pipeline")
    st.code(
        "CSV bruts (REES46, octobre 2019, échantillon de 10 % des clients)\n"
        "   -> Spark : ingestion en Parquet partitionné par jour        (Bronze)\n"
        "   -> Spark : validation, déduplication, enrichissement        (Silver)\n"
        "   -> Spark : agrégats, RFM, variables client                  (Gold)\n"
        "   -> Spark MLlib : K-Means, Random Forest / GBT, ALS\n"
        "   -> Streamlit : tableau de bord", language="text")
    st.caption("Source : jeu de données public REES46 (Kaggle). Données anonymisées, aucun traitement de données personnelles réelles.")


PAGES = {
    "Vue d'ensemble": page_overview,
    "Parcours client": page_funnel,
    "Segments": page_segments,
    "Prédiction et ciblage": page_prediction,
    "Recommandation": page_reco,
    "Streaming (démo)": page_streaming,
    "Données et architecture": page_data,
}
st.sidebar.title("Comportement client")
PAGES[st.sidebar.radio("Navigation", list(PAGES))]()
st.sidebar.caption("Projet Big Data : Spark, Parquet, MLlib, Streamlit")
