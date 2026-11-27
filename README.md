# Analyse du comportement client à grande échelle

Projet de fin de module **Big Data** : une chaîne complète de traitement et d'analyse de logs e-commerce
(ingestion, nettoyage, analyse du parcours client, segmentation, prédiction d'achat, recommandation,
traitement en continu, tableau de bord, benchmarks), construite avec **Apache Spark**, **Parquet**,
**Spark MLlib**, **Spark Structured Streaming** et **Streamlit**.

Dépôt : <https://github.com/soufianetadlaoui/bigdata-customer-behavior>

## Sommaire

- [1. Le projet en bref](#1-le-projet-en-bref)
- [2. Les données](#2-les-données)
- [3. Architecture et fonctionnement](#3-architecture-et-fonctionnement)
- [4. Structure du dépôt](#4-structure-du-dépôt)
- [5. Installation](#5-installation)
- [6. Exécution pas à pas](#6-exécution-pas-à-pas)
- [7. Fonctionnement détaillé de chaque étape](#7-fonctionnement-détaillé-de-chaque-étape)
- [8. Résultats des benchmarks](#8-résultats-des-benchmarks)
- [9. Où trouver les résultats](#9-où-trouver-les-résultats)
- [10. Limites et précautions d'interprétation](#10-limites-et-précautions-dinterprétation)
- [11. Dépannage](#11-dépannage)
- [12. Ce qui est versionné dans Git](#12-ce-qui-est-versionné-dans-git)
- [13. Pistes d'amélioration](#13-pistes-damélioration)
- [14. Glossaire](#14-glossaire)
- [15. Données, éthique, auteur](#15-données-éthique-auteur)
- [Annexe : fichiers de configuration](#annexe--fichiers-de-configuration)

---

## 1. Le projet en bref

### 1.1 Problème traité

Un site e-commerce enregistre chaque action de ses visiteurs (page vue, ajout au panier, achat). Ces journaux
sont volumineux, rapides à croître et peu exploitables tels quels. Le projet répond à la question suivante :

> Comment concevoir et mettre en œuvre une plateforme Big Data capable d'ingérer, de traiter et d'analyser des
> millions d'événements de navigation et d'achat, afin de comprendre les clients, de prédire leur comportement
> et de fournir des indicateurs exploitables pour le marketing ?

### 1.2 Ce que fait le projet

| Domaine | Ce qui est produit |
|---|---|
| Ingestion | CSV brut converti en Parquet partitionné par jour (couche Bronze) |
| Qualité | 6 règles de validation, lignes rejetées comptées et conservées (couche Silver) |
| Analyse du parcours | Entonnoir vue → panier → achat, conversion par catégorie et par prix, cohortes, délais |
| Segmentation | 6 segments d'acheteurs par RFM + K-Means, plus 2 groupes de non-acheteurs |
| Prédiction | Probabilité d'achat dans les 10 jours (régression logistique, Random Forest, Gradient Boosting) |
| Recommandation | Top 10 de produits par client (ALS), comparé à deux méthodes de référence |
| Temps réel | Flux simulé, indicateurs par minute, détection de paniers sans achat à 10 minutes |
| Restitution | Tableau de bord Streamlit à 7 pages |
| Performance | Benchmarks : formats, partitionnement, volume, nombre de cœurs, comparaison avec pandas |

### 1.3 Résultats principaux (mesurés sur la machine de référence)

| Résultat | Valeur |
|---|---|
| Événements traités | 4 224 899 (10 % des clients d'octobre 2019), 4 214 874 après nettoyage (99,76 %) |
| Clients / sessions / produits | 301 632 / 919 500 / 115 446 |
| Acheteurs | 34 792, soit **11,5 %** des clients |
| Chiffre d'affaires | 23 392 327 (unité du jeu de données), environ 672 par acheteur |
| Abandon de panier (client × produit, sur le mois) | **50,3 %** |
| Achats sans panier préalable | **55,3 %** des produits achetés |
| Concentration de la valeur | 2 segments (15 % des acheteurs) génèrent **57 %** du chiffre d'affaires |
| Prédiction d'achat à 10 jours | AUC-ROC **0,78 à 0,79**, lift ×4,6 sur les 10 % les mieux notés |
| Recommandation (HitRate@10) | Reciblage 31,2 % > ALS 27,6 % > produits populaires 19,7 % |
| Streaming | 471 paniers classés : 30,1 % convertis en 10 min, 69,9 % déclenchent une alerte |
| CSV → Parquet | Taille divisée par 5,2, requêtes 2,5 à 5 fois plus rapides |

> Ces chiffres sont des résultats de ce projet, sur un seul mois et un échantillon de clients. Lisez la
> section [Limites et précautions d'interprétation](#10-limites-et-précautions-dinterprétation) avant de les citer.

### 1.4 État d'avancement par rapport au cahier des charges

| Objectif | Énoncé | Statut | Remarque |
|---|---|---|---|
| O1 | Ingérer plus de 50 M d'événements | **Partiel** | 4,2 M traités (échantillon de 10 % des clients). Le CSV complet (≈ 42 M) dépasse la RAM de la machine de référence |
| O2 | Nettoyer de façon distribuée, règles documentées | Atteint | 6 règles, rapport JSON |
| O3 | Analyser l'entonnoir de conversion | Atteint | Au niveau session et au niveau client × produit |
| O4 | 4 à 6 segments, silhouette > 0,4 | **Partiel** | 6 segments interprétables, mais silhouette de 0,38 |
| O5 | AUC > 0,80 | **Non atteint** | 0,78 à 0,79 |
| O6 | Recommandation avec Precision@10 et Recall@10 | Atteint | ALS bat les produits populaires, pas le reciblage |
| O7 | Démontrer la scalabilité | **Partiel** | Mesures réalisées, mais coûts fixes dominants à cette échelle |
| O8 | Tableau de bord d'au moins 10 indicateurs | Atteint | 7 pages |

Éléments du cahier des charges non réalisés : orchestration **Airflow** (le dossier `dags/` est vide),
**Kafka** (remplacé par un dossier surveillé), base de service MongoDB/Cassandra (remplacée par des fichiers
Parquet et CSV), déploiement Kubernetes.

---

## 2. Les données

### 2.1 Source

Jeu de données public **« eCommerce behavior data from multi-category store »**, publié par REES46 sur Kaggle.
Il contient les événements d'un magasin en ligne, un fichier par mois. Ce projet utilise
**`2019-Oct.csv`** (environ 5,4 Go, ≈ 42 M de lignes).

Téléchargement manuel (compte Kaggle nécessaire) : placez le fichier dans `data/raw/2019-Oct.csv`.
Il n'est pas versionné dans Git.

### 2.2 Schéma d'origine

| Colonne | Description |
|---|---|
| `event_time` | Horodatage UTC (format `2019-10-01 00:00:00 UTC`) |
| `event_type` | `view`, `cart`, `remove_from_cart`, `purchase` |
| `product_id`, `category_id` | Identifiants produit et catégorie |
| `category_code` | Catégorie hiérarchique (ex. `electronics.smartphone`), **vide dans environ 32 % des événements** |
| `brand` | Marque (parfois vide) |
| `price` | Prix du produit |
| `user_id`, `user_session` | Identifiants client et session |

### 2.3 Échantillonnage

Le traitement du fichier complet dépasse les capacités de la machine de référence (8 Go de RAM). Le projet
travaille donc sur **10 % des clients** : `make_sample.py` garde toutes les lignes dont `user_id % 10 == 0`.
Ce choix conserve l'**historique complet** de chaque client retenu (indispensable pour la récence, la
fréquence et la prédiction), contrairement à un échantillon des premières lignes ou des lignes au hasard.

---

## 3. Architecture et fonctionnement

### 3.1 Vue d'ensemble

```
                       ┌──────────────────────── Spark (mode local, Docker) ───────────────────────┐
CSV brut ─ échantillon ─► BRONZE ─────────► SILVER ─────────► GOLD                                   │
2019-Oct.csv  10 %      Parquet partitionné  Validé, dédoublonné  Agrégats et tables client          │
                        par jour             enrichi              (segments, scores, recommandations)│
                                                  │                        ▲                         │
                                                  ├──► Analyse du parcours (CSV de résultats)        │
                                                  ├──► RFM + K-Means ──────┤                         │
                                                  ├──► Prédiction d'achat ─┤  (MLlib)                │
                                                  └──► ALS (recommandation)┘                         │
                       └────────────────────────────────────────────────────────────────────────────┘
                                                  │
Silver ─► producteur ─► dossier de fichiers JSON ─► Structured Streaming ─► indicateurs + alertes paniers
                                                  │
                         docs/results/*.csv + data/gold/* + data/stream/out/* ─► Tableau de bord Streamlit
```

### 3.2 Les couches de données

| Couche | Emplacement | Format | Contenu |
|---|---|---|---|
| Raw | `data/raw/` | CSV | Fichiers sources et échantillon |
| Bronze | `data/bronze/events/` | Parquet, partitionné par `event_date` | Données brutes typées (dates converties), colonnes `ingestion_ts` et `source_file` |
| Silver | `data/silver/events/` | Parquet, partitionné par `event_date` | Données validées, sans doublons, enrichies |
| Gold | `data/gold/` | Parquet | Tables prêtes à l'usage : `customer_segments`, `purchase_scores`, `recommendations`, `product_catalog` |
| Résultats | `docs/results/` | CSV, JSON | Petits agrégats pour le rapport et le tableau de bord (versionnés) |

### 3.3 Choix techniques et justifications

| Choix | Raison |
|---|---|
| **Spark en mode local** (`local[4]`) pour les calculs | L'ingestion du CSV complet en grappe (1 maître + 2 workers) n'a pas abouti avec 8 Go de RAM. Le mode local donne les mêmes résultats avec bien moins de mémoire |
| **Parquet** | Format colonnaire compressé : 5,2 fois plus petit que le CSV et requêtes 2,5 à 5 fois plus rapides (voir benchmarks) |
| **Partitionnement par jour** | Permet de ne lire que les jours utiles ; gain mesuré faible à cette échelle (voir benchmarks) |
| **Échantillon de 10 % des clients** | Seul moyen de tenir dans 8 Go tout en gardant des historiques clients complets |
| **MinIO abandonné** | L'image `minio/minio` n'est plus disponible sur Docker Hub ; Spark écrit directement dans `data/` (partagé avec les conteneurs) |
| **Dossier surveillé au lieu de Kafka** | Kafka est trop lourd pour la machine ; la logique de traitement (fenêtres, filigranes, jointure flux-flux) est la même, seul le connecteur d'entrée changerait |
| **Séparation temporelle pour le machine learning** | Un découpage aléatoire mélangerait passé et futur et gonflerait les scores |
| **Streamlit en dehors de Docker** | Il lit directement les fichiers du dépôt et se lance en une commande |

### 3.4 Machine de référence

| Élément | Valeur |
|---|---|
| Processeur | Intel Core i7-1165G7 (4 cœurs, 8 threads) |
| RAM | 8 Go (7,7 Go utilisables) ; WSL2 limité à 5 Go et 4 processeurs |
| Logiciels | Windows, Docker Desktop (WSL2), Spark 3.5.1, Java 11, Python 3.10+ |

Les durées indiquées plus bas sont celles de cette machine.

---

## 4. Structure du dépôt

```
bigdata-customer-behavior/
├── README.md                     Ce fichier
├── requirements.txt              Dépendances Python (environnement Windows)
├── docker-compose.yml            Spark (1 maître + 2 workers)
├── docker/spark/Dockerfile       Image Spark avec numpy (nécessaire à MLlib)
├── .env.example                  Variables d'environnement (copier en .env)
├── .gitignore
├── Makefile                      Raccourcis Docker
├── data/                         Données (ignorées par Git, sauf les .gitkeep)
│   ├── raw/  bronze/  silver/  gold/  stream/
├── docs/
│   ├── quality_report.json       Rapport qualité (étape 3)
│   ├── results/                  Agrégats CSV (versionnés)
│   └── images/                   Graphiques 01 à 20 (versionnés)
├── src/
│   ├── ingestion/    make_sample.py, csv_to_parquet.py
│   ├── processing/   bronze_to_silver.py, eda_funnel.py, funnel_product.py, make_charts.py
│   ├── ml/           rfm_segmentation.py, purchase_prediction.py, recommender_als.py,
│   │                 make_segment_charts.py, make_prediction_charts.py, make_reco_charts.py
│   ├── streaming/    stream_producer.py, streaming_demo.py
│   └── utils/        spark_session.py
├── dashboard/        app.py (Streamlit)
├── benchmarks/       benchmark.py, bench_pandas.py, make_benchmark_charts.py
├── tests/            test_spark_smoke.py (test de fumée, nécessite PySpark et Java en local)
├── dags/  notebooks/ Dossiers réservés, non utilisés
```

---

## 5. Installation

### 5.1 Prérequis

- Windows 10/11 avec **Docker Desktop** (moteur WSL2), Git, et **Python 3.10 ou supérieur**.
- 8 Go de RAM minimum, 15 Go d'espace disque libre.
- Un compte Kaggle pour télécharger le jeu de données.

### 5.2 Limiter la mémoire de WSL2 (recommandé avec 8 Go)

Sans limite, WSL2 peut réserver trop de RAM et faire tomber Docker. Créez le fichier
`C:\Users\<vous>\.wslconfig` :

```powershell
@"
[wsl2]
memory=5GB
processors=4
swap=4GB
"@ | Out-File -Encoding ascii "$env:USERPROFILE\.wslconfig"
wsl --shutdown
```

Relancez ensuite Docker Desktop. Vérification : `docker run --rm alpine free -m` doit afficher environ 4,9 Go.

### 5.3 Récupérer le projet et l'environnement Python

```powershell
git clone https://github.com/soufianetadlaoui/bigdata-customer-behavior.git
cd bigdata-customer-behavior
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si l'activation est bloquée : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 5.4 Construire et démarrer Spark

```powershell
Copy-Item .env.example .env
docker compose build
docker compose up -d
docker compose ps
```

Vérifications :
- <http://localhost:8080> : interface du maître Spark (2 workers « ALIVE » si la grappe est active).
- Pour économiser la RAM, les calculs lourds s'exécutent en **mode local** dans le conteneur `spark-master` :
  vous pouvez arrêter les workers avec `docker compose stop spark-worker-1 spark-worker-2`.

L'image personnalisée (`docker/spark/Dockerfile`) ajoute **numpy**, indispensable à `pyspark.ml`
(K-Means, régression, ALS). Les fichiers de configuration sont donnés en [annexe](#annexe--fichiers-de-configuration).

### 5.5 Placer les données

Téléchargez `2019-Oct.csv` depuis Kaggle et copiez-le dans `data\raw\`.

> Terminal PowerShell : pour éviter des caractères déformés (`├®`) dans les messages, tapez
> `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` avant les commandes.

---

## 6. Exécution pas à pas

Toutes les commandes se lancent **depuis la racine du dépôt**, dans PowerShell, avec le `.venv` activé
et Docker démarré. Dans les conteneurs, le dépôt est monté sous `/opt/project`.

Pour masquer les messages Spark verbeux, ajoutez `2>&1 | Select-String -NotMatch " INFO "` en fin de
commande. **Un script Spark peut sembler figé : c'est normal** (les `print` ne s'affichent qu'à la fin).
Suivez l'avancement sur <http://localhost:4040> (onglet *Jobs*) pendant l'exécution.

| Étape | Script | Où | Durée mesurée |
|---|---|---|---|
| 1 | `make_sample.py` | Windows | quelques minutes |
| 2 | `csv_to_parquet.py` | Docker | 37 s |
| 3 | `bronze_to_silver.py` | Docker | ≈ 5 min |
| 4 | `eda_funnel.py`, `funnel_product.py`, `make_charts.py` | Docker, Windows | 3 à 6 min (estimation), 35 s |
| 5 | `rfm_segmentation.py`, `make_segment_charts.py` | Docker, Windows | 2 à 3 min |
| 6a | `purchase_prediction.py`, `make_prediction_charts.py` | Docker, Windows | ≈ 6 min |
| 6b | `recommender_als.py`, `make_reco_charts.py` | Docker, Windows | ≈ 7 min |
| 7a | `dashboard/app.py` | Windows | immédiat |
| 7b | `streaming_demo.py`, `stream_producer.py` | Docker, Windows | ≈ 7 min |
| 8 | `benchmark.py`, `bench_pandas.py`, `make_benchmark_charts.py` | Docker, Windows | ≈ 30 min |

### Étape 1 : créer l'échantillon de 10 % des clients

```powershell
python src\ingestion\make_sample.py
```
Lit `data\raw\2019-Oct.csv` par blocs d'un million de lignes (peu de mémoire) et écrit
`data\raw\oct_sample10.csv`.
**Vérification :** la dernière ligne affichée est `4,224,899 lignes conservées`.

### Étape 2 : ingestion en Parquet (Bronze)

```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/ingestion/csv_to_parquet.py --input /opt/project/data/raw/oct_sample10.csv --output /opt/project/data/bronze/events
```
Option de test rapide : `--sample 0.01` (1 % des lignes).
**Vérification :** 4 224 899 lignes, Parquet d'environ 104 Mo (CSV de 542 Mo), répartition
≈ 96 % de vues, 2,2 % de paniers, 1,7 % d'achats.

### Étape 3 : nettoyage (Silver)

```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/processing/bronze_to_silver.py
```
Écrit `data/silver/events`, `data/silver/rejected` (lignes rejetées, audit) et `docs/quality_report.json`.
Option : `--max-session-events 500` (seuil de détection des bots).
**Vérification :** Silver = 4 214 874 lignes (99,76 %).

### Étape 4 : analyse du parcours client

```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/processing/eda_funnel.py
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/processing/funnel_product.py
python src\processing\make_charts.py
```
Écrit 12 CSV dans `docs/results/` et les graphiques 01 à 07 dans `docs/images/`.
**Vérification :** KPI « Clients : 301 632 | Sessions : 919 500 », « Abandon de panier : 50,3 % ».

### Étape 5 : segmentation RFM + K-Means

Première exécution : numérotation automatique des segments, par chiffre d'affaires décroissant.
```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/ml/rfm_segmentation.py
```
Lisez le profil de chaque segment, puis **nommez-les** et relancez (les noms sont appliqués dans l'ordre du
chiffre d'affaires décroissant ; évitez les accents dans l'argument, PowerShell peut les déformer) :
```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/ml/rfm_segmentation.py --k 6 --names "VIP multi-categories,Fideles forte valeur,Ponctuels inactifs,Nouveaux acheteurs,Reguliers explorateurs,Petits paniers navigateurs"
python src\ml\make_segment_charts.py
```
Écrit `data/gold/customer_segments` et les graphiques 08 à 11.
**Vérification :** « Clients : 301 632 (acheteurs : 34 792) », `k retenu : 6`.

### Étape 6a : prédiction de l'achat à 10 jours

```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/ml/purchase_prediction.py
python src\ml\make_prediction_charts.py
```
Écrit `data/gold/purchase_scores` et les graphiques 12 à 14.
**Vérification :** entraînement 140 977 clients (5,40 % achètent), test 230 738 clients (3,58 %), AUC-ROC ≈ 0,78 à 0,79.

### Étape 6b : recommandation de produits (ALS)

```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/ml/recommender_als.py
python src\ml\make_reco_charts.py
```
Écrit `data/gold/recommendations`, `data/gold/product_catalog` et le graphique 15.
**Vérification :** 8 259 clients évalués ; HitRate@10 d'environ 19,7 % (populaires), 31,2 % (reciblage), 27,6 % (ALS).

### Étape 7a : tableau de bord

```powershell
streamlit run dashboard/app.py
```
Ouvre <http://localhost:8501>. Les pages « Prédiction et ciblage » et « Recommandation » (fiche client)
lisent `data/gold/` : elles nécessitent les étapes 5, 6a et 6b.

### Étape 7b : traitement en continu (3 terminaux)

Le **terminal A** démarre Spark (toujours en premier, avec `--reset`) :
```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/src/streaming/streaming_demo.py --reset
```
Le **terminal B** rejoue les événements (`--clean` vide le dossier d'entrée) :
```powershell
python src\streaming\stream_producer.py --clean
```
Le **terminal C** : `streamlit run dashboard/app.py`, page « Streaming (démo) », bouton « Rafraîchir ».

Par défaut : 15 octobre 2019, de 14 h à 15 h 30 UTC, un fichier (1 minute d'activité) toutes les 2 secondes.
Spark s'arrête seul après 90 secondes sans nouvelle donnée et affiche un bilan.
Les premières alertes arrivent environ 11 minutes d'activité après le début (20 à 25 secondes de démonstration).
**Pour rejouer :** toujours `--reset` côté Spark et `--clean` côté producteur.

### Étape 8 : benchmarks

Fermez les applications lourdes et libérez de la RAM :
```powershell
docker compose stop spark-worker-1 spark-worker-2
```
Exécutez ensuite, dans cet ordre (une expérience à la fois, jamais en parallèle) :
```powershell
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/benchmarks/benchmark.py --experiment formats
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/benchmarks/benchmark.py --experiment partitioning
docker exec spark-master /opt/spark/bin/spark-submit --master "local[4]" --driver-memory 3g /opt/project/benchmarks/benchmark.py --experiment datasize
foreach ($n in 1,2,4) { docker exec spark-master /opt/spark/bin/spark-submit --master "local[$n]" --driver-memory 3g /opt/project/benchmarks/benchmark.py --experiment cores }
```
Puis, **Docker arrêté** pour libérer la mémoire (la référence pandas en a besoin) :
```powershell
docker compose down
wsl --shutdown
python benchmarks\bench_pandas.py
python benchmarks\make_benchmark_charts.py
```
Écrit `docs/results/benchmark.csv`, `benchmark_env.json` et les graphiques 16 à 20.
**Option** (après avoir relancé `docker compose up -d spark-master`) : `--experiment fullcsv` lit le CSV complet
de 5,4 Go en mode local. Non testée sur la machine de référence, elle peut faire planter Docker : ce serait
alors un résultat à documenter.

### Sauvegarder son travail

Après chaque étape :
```powershell
git add .
git commit -m "Etape N : description"
git push
```
Les résultats légers (`docs/results`, `docs/images`) sont versionnés ; les données (`data/`) ne le sont pas.

---

## 7. Fonctionnement détaillé de chaque étape

### 7.1 Ingestion (Bronze)

Lecture du CSV avec un **schéma explicite** (plus rapide et plus sûr que l'inférence), conversion de
`event_time` en horodatage, ajout de `event_date` (colonne de partition), `ingestion_ts` et `source_file`.
Écriture en Parquet compressé (snappy), **partitionné par jour**.

### 7.2 Nettoyage (Silver)

| Règle | Contrôle | Lignes rejetées |
|---|---|---|
| R1 | Champs obligatoires nuls (`event_time`, `event_type`, `user_id`, `product_id`) | 0 |
| R2 | Type d'événement invalide | 0 |
| R3 | Prix nul, négatif ou manquant | 6 808 (0,16 %) |
| R4 | `user_session` manquante | 0 |
| R5 | Doublons exacts (heure, type, produit, client, session) | 3 217 (0,08 %) |
| R6 | Sessions de plus de 500 événements (bots probables) | 0 |

Le seuil R6 n'est jamais atteint : la plus grande session compte 288 événements (médiane : 2 ; 99ᵉ percentile : 31).
La règle reste en place comme garde-fou.
Enrichissements : `category_main`, `category_sub` (extraits de `category_code`), `hour`, `day_of_week`,
`is_weekend`, `part_of_day`. Les marques et catégories manquantes deviennent `unknown` (non imputées : 32 % des
événements ont une catégorie inconnue, à traiter à part dans toute analyse par catégorie).

Colonnes Silver : `user_id`, `user_session`, `event_type`, `event_time`, `product_id`, `category_id`,
`category_code`, `brand`, `price`, `event_date`, `category_main`, `category_sub`, `hour`, `day_of_week`,
`is_weekend`, `part_of_day`.

### 7.3 Analyse du parcours client

Deux entonnoirs, volontairement distincts :

- **Au niveau session** (`eda_funnel.py`) : un achat peut survenir dans une session sans événement « panier »
  (≈ 33 000 cas). L'entonnoir n'est alors pas monotone (plus de sessions d'achat que de sessions avec panier).
- **Au niveau client × produit** (`funnel_product.py`) : on suit un produit d'une session à l'autre sur le mois,
  ce qui garantit un entonnoir monotone. **C'est celui à citer** : 2 319 777 produits consultés, 50 075 mis au
  panier (2,16 %), 24 898 achetés après panier (1,07 %), abandon de panier 50,3 %.

D'autres résultats : conversion par catégorie (electronics ≈ 8,6 %, appliances ≈ 6,3 %, apparel ≈ 1,9 %), par
tranche de prix, heatmap jour × heure, cohortes hebdomadaires, délai panier → achat **dans une même session**
(médiane 1,5 min, 90 % en moins de 5,4 min), acheteurs selon le nombre d'articles achetés.

### 7.4 Segmentation RFM + K-Means

- **Population de clustering :** les 34 792 acheteurs. Variables : récence, fréquence (nombre de **sessions d'achat**
  distinctes, pas d'articles), montant, nombre de sessions, diversité des produits (`category_id`).
- Transformation `log1p` des variables asymétriques puis standardisation.
- **Choix de k :** K-Means pour k de 2 à 8, coude (WSSSE) et silhouette ; k retenu parmi 3 à 6 par silhouette
  maximale (k = 6, silhouette 0,3825).
- **Non-acheteurs** (règle simple) : « Panier sans achat » (13 424) et « Visiteurs » (253 416).
- Les segments sont numérotés par chiffre d'affaires décroissant et **nommés à la main** d'après leur profil.

| Segment | Clients | Récence (j) | Fréquence | Montant | Part du CA |
|---|---|---|---|---|---|
| VIP multi-catégories | 1 394 | 6,3 | 9,4 | 4 789 | 28,5 % |
| Fidèles forte valeur | 3 940 | 10,8 | 2,95 | 1 679 | 28,3 % |
| Ponctuels inactifs | 8 920 | 23,7 | 1,09 | 362 | 13,8 % |
| Nouveaux acheteurs | 9 292 | 7,4 | 1,08 | 324 | 12,9 % |
| Réguliers explorateurs | 4 355 | 11,2 | 2,5 | 678 | 12,6 % |
| Petits paniers navigateurs | 6 891 | 15,9 | 1,11 | 131 | 3,9 % |

### 7.5 Prédiction de l'achat à 10 jours

**Question :** un client actif jusqu'à une date donnée achètera-t-il dans les 10 jours suivants ?

| | Variables calculées sur | Cible |
|---|---|---|
| Entraînement | événements jusqu'au 11 octobre | achat du 12 au 21 octobre |
| Test | événements jusqu'au 21 octobre | achat du 22 au 31 octobre |

La cible d'entraînement s'arrête à la date de coupure du test : le modèle ne voit jamais les achats de la
période de test. **22 variables** : volumes d'activité, sessions, jours actifs, produits et catégories vus,
fréquence et montant d'achat, activité des 3 et 7 derniers jours, récence, ancienneté, jours depuis le dernier achat,
indicateur de **panier ouvert** (ajout au panier plus récent que le dernier achat), taux de panier, etc.

| Modèle | AUC-ROC | AUC-PR | Précision top 10 % | Rappel top 10 % | Lift |
|---|---|---|---|---|---|
| Régression logistique | 0,7807 | 0,2100 | 16,3 % | 45,9 % | 4,55 |
| Random Forest | 0,7856 | 0,2244 | 16,5 % | 46,5 % | 4,60 |
| Gradient Boosting (modèle final) | 0,7867 | 0,2123 | 16,4 % | 46,1 % | 4,57 |

Taux d'achat de base du test : 3,58 % (AUC-PR d'un modèle aléatoire : 0,036). Les 20 % de clients les mieux
notés captent **63 %** des acheteurs. Variable la plus importante : le montant déjà dépensé (`monetary`, 0,26).
Le modèle final est choisi **à l'avance**, pas d'après le score de test.

### 7.6 Recommandation (ALS)

- **Retour implicite :** vue = 1, panier = 3, achat = 5, puis `log1p` de la somme par couple (client, produit).
- **Modèle :** ALS (rang 32, régularisation 0,05, alpha 10, 10 itérations) sur 1 621 345 interactions,
  230 738 clients, 100 537 produits (densité 0,007 %).
- **Évaluation temporelle :** apprentissage jusqu'au 21 octobre ; vérité = produits achetés du 22 au 31 octobre
  par 8 259 clients connus.
- **Deux références :** produits les plus achetés (liste unique) et produits récemment consultés (reciblage).

| Méthode | Precision@10 | Recall@10 | HitRate@10 |
|---|---|---|---|
| Produits les plus achetés | 2,24 % | 15,86 % | 19,69 % |
| Produits récemment consultés | 3,53 % | 25,83 % | **31,19 %** |
| ALS | 3,25 % | 23,08 % | 27,62 % |

Le reciblage bat ALS : beaucoup d'achats portent sur un produit déjà consulté. ALS apporte surtout de la
découverte et bat nettement la liste unique.

### 7.7 Traitement en continu

1. **Source :** `stream_producer.py` rejoue une tranche horaire du Silver sous forme de fichiers JSON
   (1 fichier = 1 minute), écrits sous un nom temporaire puis renommés (Spark ne lit jamais un fichier incomplet).
2. **Indicateurs par minute :** fenêtre d'une minute (événements par type, chiffre d'affaires, clients actifs)
   avec filigrane (*watermark*) d'une minute.
3. **Paniers sans achat :** **jointure externe flux-flux** entre les ajouts au panier et les achats du même produit
   par le même client dans les 10 minutes suivantes. Un panier sans achat devient une **alerte**, enrichie du
   segment RFM du client (jointure avec la table Gold).
4. **Temps d'événement :** les fenêtres suivent l'heure des événements (octobre 2019), pas celle de la machine.
   Une alerte n'apparaît que lorsque le filigrane dépasse l'heure du panier + 10 minutes.

Résultat de la démonstration : 16 224 événements, 495 paniers ; 471 classés (142 convertis en 10 min, 30,1 % ;
329 alertes, 69,9 %) ; les 24 restants attendent l'avancée du filigrane.

### 7.8 Tableau de bord

| Page | Contenu | Données |
|---|---|---|
| Vue d'ensemble | KPI, activité quotidienne, heatmap jour × heure | `docs/results` |
| Parcours client | Entonnoir client × produit, conversion par catégorie et par prix, délai panier → achat | `docs/results` |
| Segments | Poids clients vs chiffre d'affaires, profils, nuage récence × montant, choix de k | `docs/results` |
| Prédiction et ciblage | Comparaison des modèles, déciles, importance des variables, **outil de ciblage** avec export CSV | `docs/results`, `data/gold/purchase_scores` |
| Recommandation | Comparaison des méthodes, **fiche client** | `docs/results`, `data/gold/*` |
| Streaming (démo) | Indicateurs par minute, alertes par segment | `data/stream/out` |
| Données et architecture | Lignes rejetées par règle, schéma du pipeline | `docs/quality_report.json` |

### 7.9 Benchmarks

Méthode : une exécution de chauffe, puis 3 mesures, **médiane** retenue (plage min–max conservée). Les
requêtes sont exécutées jusqu'au bout sans écriture sur disque, et les temps **incluent la lecture des fichiers**.

| Expérience | Ce qui est comparé |
|---|---|
| `formats` | CSV (schéma explicite ou inféré) contre Parquet, 3 requêtes |
| `partitioning` | Parquet partitionné par jour contre copie non partitionnée |
| `datasize` | Même traitement sur 10, 25, 50 et 100 % des clients |
| `cores` | Même traitement avec 1, 2 et 4 cœurs |
| `fullcsv` | CSV complet de 5,4 Go (option, non exécutée) |
| pandas | Même agrégation sur une seule machine |

---

## 8. Résultats des benchmarks

| Mesure | Résultat |
|---|---|
| Taille CSV / Parquet | 568 Mo / 109 Mo (5,2 fois) |
| Comptage par type, CSV / Parquet | 8,7 s / 2,1 s |
| Prix moyen des achats, CSV / Parquet | 4,7 s / 1,9 s (2,5 fois) |
| Agrégation par client, CSV / Parquet | 20,3 s (plage 12,0 à 24,3) / 4,1 s (5,0 fois) |
| Coût de l'inférence de schéma | environ +6 s |
| Partitionné / non partitionné, 1 jour | 1,70 s / 1,27 s (pas de gain à cette échelle) |
| Volume ×9,9 (426 k → 4,2 M lignes) | temps +17 % seulement (10,2 s → 11,9 s) |
| Accélération, agrégation par client | ×1,73 (2 cœurs), ×2,04 (4 cœurs) |
| Accélération, entonnoir par session | ×1,84 (2 cœurs), ×2,26 (4 cœurs) |
| pandas (1 cœur, CSV) | 14,6 s (lecture 4,4 s, DataFrame de 735 Mo) |

Interprétation : le Parquet est nettement meilleur ; à cette échelle, les **coûts fixes de Spark (≈ 10 s)**
dominent, ce qui explique la quasi-absence de gain avec le volume, l'absence d'effet du partitionnement et la
saturation au-delà de 2 cœurs. pandas reste compétitif sur le même CSV : à 4,2 M de lignes, une seule machine
suffit. Spark devient indispensable quand les données dépassent la mémoire (le CSV complet représenterait
environ 7 Go en DataFrame pandas, estimation non mesurée).

---

## 9. Où trouver les résultats

| Fichier produit | Script |
|---|---|
| `kpis`, `daily`, `hourly`, `funnel`, `time_to_purchase`, `funnel_category`, `funnel_price_band`, `top_brands`, `top_products`, `cohorts`, `repeat_buyers` (`docs/results/*.csv`) | `eda_funnel.py` |
| `funnel_product.csv` | `funnel_product.py` |
| `k_selection`, `segment_profiles`, `segment_sizes`, `segment_sample` | `rfm_segmentation.py` |
| `purchase_model_metrics`, `purchase_feature_importance`, `purchase_deciles` | `purchase_prediction.py` |
| `recommender_metrics.csv` | `recommender_als.py` |
| `docs/quality_report.json` | `bronze_to_silver.py` |
| `benchmark.csv`, `benchmark_env.json` | `benchmark.py`, `bench_pandas.py` |
| `data/gold/customer_segments` | `rfm_segmentation.py` |
| `data/gold/purchase_scores` | `purchase_prediction.py` |
| `data/gold/recommendations`, `data/gold/product_catalog` | `recommender_als.py` |
| `data/stream/out/*` | `streaming_demo.py` |

Graphiques (`docs/images/`) : 01 évolution quotidienne, 02 activité heure × jour, 03 entonnoir, 04 conversion par
catégorie, 05 conversion par prix, 06 rétention par cohorte, 07 top marques, 08 choix de k, 09 taille des segments,
10 profil des segments, 11 nuage des segments, 12 comparaison des modèles, 13 lift par décile, 14 importance
des variables, 15 comparaison des recommandations, 16 formats, 17 partitionnement, 18 volume, 19 cœurs, 20 pandas contre Spark.

---

## 10. Limites et précautions d'interprétation

À écrire noir sur blanc dans le rapport et à la soutenance :

1. **Un seul mois, 10 % des clients.** Pas de saisonnalité, rétention observée sur quelques semaines seulement.
2. **Silhouette de 0,38 (< 0,4).** Les segments sont interprétables, mais les comportements forment un continuum.
3. **Segments calculés sur tout le mois.** En enrichissant des alertes de streaming du 15 octobre, on utilise une
   information « du futur » : 37 % des alertes concernent le segment « Panier sans achat », qui n'achète
   jamais en octobre par définition. Acceptable pour une démonstration, pas pour la production.
4. **55 % des produits achetés n'ont aucun événement « panier » préalable** : cause inconnue (achat direct,
   panier créé avant le 1ᵉʳ octobre, lacune de traçage). L'entonnoir ne décrit que le chemin via le panier.
5. **Les « achats » sont des articles, pas des commandes.** Le nombre d'acheteurs récurrents est une borne haute ;
   le prix moyen par article (≈ 317) n'est pas le panier moyen (≈ 371 par session d'achat).
6. **Le score d'achat sert à classer, pas à estimer une probabilité.** Le taux de base passe de 5,4 % (entraînement)
   à 3,6 % (test) ; les probabilités surestimeraient le taux réel. Le modèle prédit une propension, pas l'effet
   d'une campagne (pas d'uplift).
7. **Les trois modèles se valent** (écart d'AUC inférieur à 0,01, sans intervalle de confiance).
8. **ALS ne bat pas le reciblage.** Les produits déjà consultés ne sont pas exclus des recommandations.
9. **Alerte de streaming ≠ abandon définitif.** Environ 28 % des alertes (estimation grossière) concerneraient
   des clients qui achèteraient plus tard. Le seuil de 10 minutes est réglable (`--abandon-minutes`).
10. **Délai panier → achat (médiane 1,5 min) valable dans une même session uniquement.**
11. **Benchmarks :** les mesures CSV sont instables (plage jusqu'à ×2) ; le cache disque du système reste actif ;
    la copie « non partitionnée » est mélangée aléatoirement (la comparaison de taille 109 Mo / 252 Mo est donc
    biaisée) ; la machine ne permet pas de conclure à une scalabilité linéaire.
12. **Reproductibilité :** tous les résultats et durées proviennent d'une seule machine (8 Go de RAM). Les temps
    varient d'une machine à l'autre ; les graines aléatoires sont fixées (K-Means, modèles, ALS), mais de légères
    variations de scores restent possibles.

---

## 11. Dépannage

| Symptôme | Cause et solution |
|---|---|
| `pull access denied for minio/minio` | MinIO n'est plus sur Docker Hub : il n'est pas utilisé dans ce projet, retirez-le du `docker-compose.yml` |
| Docker Desktop : `daemon.json is invalid` | Fermez Docker, `wsl --shutdown`, ouvrez `%USERPROFILE%\.docker\daemon.json`, remplacez son contenu par `{}`, relancez Docker |
| `ModuleNotFoundError: No module named 'numpy'` dans Spark | L'image n'a pas numpy : utilisez `docker/spark/Dockerfile` puis `docker compose up -d --build` (ou, en dépannage, `docker exec -u root spark-master pip install numpy`, perdu à la recréation du conteneur) |
| `http://localhost:4040` inaccessible | Le port 4040 doit être publié dans `docker-compose.yml` ; l'interface n'existe que pendant l'exécution d'un script |
| Script Spark « figé » sans message | Normal : suivez <http://localhost:4040> ou `docker stats --no-stream spark-master` (CPU de 200 à 400 %) |
| `Killed`, code 137, `OutOfMemoryError` | Mémoire insuffisante : `memory=4GB` dans `.wslconfig`, `SPARK_WORKER_MEMORY=1g`, fermez les autres applications, travaillez sur un échantillon plus petit |
| Docker tombe pendant un calcul lourd | Même cause ; relancez Docker, `docker compose up -d`, puis refaites l'étape |
| Caractères `├®` dans la console | `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` |
| Accents déformés dans `--names` | Utilisez des noms sans accents dans la commande, remettez-les dans le rapport |
| `Permission denied` à l'écriture | `docker exec -u root ...` |
| `git push` : `no upstream branch` | `git push --set-upstream origin main` |
| `git push` : `rejected ... fetch first` | `git fetch origin`, puis fusionnez ; sur un dépôt personnel, `git push --force` après vérification |
| Des fichiers de données dans Git | `.gitignore` absent ou vide : recréez-le (annexe), puis `git rm -r --cached data` |
| Streaming : alertes absentes | Elles arrivent ≈ 11 minutes d'activité après le début ; laissez Spark tourner après la fin du rejeu |
| Streaming : `Current batch is falling behind` | Avertissement sans gravité (lots de 2 à 3 s pour un déclencheur d'1 s) ; utilisez `--interval 4` côté producteur |
| Streaming : rejeu sans effet | Relancez avec `--reset` (Spark) et `--clean` (producteur) |
| Tableau de bord : « Données absentes » | Exécutez l'étape correspondante (les pages Gold exigent les étapes 5, 6a, 6b) |
| `Activate.ps1` bloqué | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## 12. Ce qui est versionné dans Git

| Versionné | Non versionné (`.gitignore`) |
|---|---|
| Code (`src/`, `dashboard/`, `benchmarks/`, `tests/`) | `data/raw`, `data/bronze`, `data/silver`, `data/gold`, `data/stream` |
| `docs/results/*.csv`, `docs/quality_report.json`, `docs/images/*.png` | `*.csv` ailleurs, `*.parquet`, `*.zip`, `*.log` |
| `docker-compose.yml`, `docker/`, `requirements.txt`, `.env.example`, `README.md` | `.env`, `.venv/`, `__pycache__/` |

Tout se **régénère** à partir de `2019-Oct.csv` en suivant la section 6.

---

## 13. Pistes d'amélioration

- **Compacter le Silver** : la même agrégation prend 11,9 s sur le Silver contre 4,1 s sur le Bronze pour des
  données équivalentes ; hypothèse à vérifier : trop de petits fichiers (`Get-ChildItem data\silver\events -Recurse -Filter *.parquet | Measure-Object`).
- **Recommandation hybride** : produits récemment consultés d'abord, complétés par ALS (règle fixée à l'avance).
- **Seuil d'alerte calibré** à partir de la distribution des délais panier → achat au niveau client × produit.
- **Référence simple pour la prédiction** (classement par montant dépensé) pour mesurer l'apport réel du modèle.
- **Volume réel** : exécuter `fullcsv` ou passer à une grappe avec plus de mémoire.
- **Orchestration Airflow** et **Kafka** (cahier des charges initial).
- **Segments calculés à date** pour un enrichissement de streaming sans information du futur.
- **Explicabilité** (valeurs SHAP) et mesure de l'**uplift** d'une campagne.

---

## 14. Glossaire

| Terme | Définition |
|---|---|
| Bronze / Silver / Gold | Couches de qualité croissante : brut, validé, prêt à l'usage |
| Parquet | Format de fichier colonnaire compressé, adapté à l'analyse |
| Partitionnement | Découpage des fichiers par valeur d'une colonne (ici le jour) |
| RFM | Récence, Fréquence, Montant : trois mesures classiques de la valeur d'un client |
| K-Means | Algorithme qui regroupe des clients similaires en k groupes |
| Silhouette | Mesure de séparation des groupes (de −1 à 1) ; plus haut = mieux séparés |
| AUC-ROC | Capacité d'un modèle à classer les acheteurs avant les non-acheteurs (0,5 = hasard) |
| AUC-PR | Comme l'AUC-ROC, plus informative quand la classe positive est rare ; référence = taux de base |
| Lift | Taux d'achat dans le groupe ciblé divisé par le taux moyen |
| ALS | *Alternating Least Squares* : filtrage collaboratif par factorisation de matrice |
| Precision@10, Recall@10, HitRate@10 | Part des 10 recommandations pertinentes ; part des achats retrouvés ; part des clients ayant au moins un succès |
| Filigrane (*watermark*) | En streaming, seuil au-delà duquel Spark considère une période d'événements comme complète |
| Jointure flux-flux | Jointure entre deux flux d'événements dans une fenêtre de temps |
| Mode local | Spark exécuté dans un seul processus, avec N fils d'exécution (`local[N]`) |

---

## 15. Données, éthique, auteur

- Les données sont publiques et anonymisées (identifiants numériques). Aucun traitement de données personnelles réelles.
  Consultez les conditions d'utilisation du jeu de données sur Kaggle avant toute réutilisation.
- La segmentation et le ciblage soulèvent des questions de vie privée et de biais (RGPD) à discuter dans le rapport.
- **Auteur :** à compléter (nom, filière, encadrant, année universitaire).

---

## Annexe : fichiers de configuration

### `docker-compose.yml`

```yaml
services:
  spark-master:
    build: ./docker/spark
    image: bigdata-spark:3.5.1
    container_name: spark-master
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master
    ports:
      - "8080:8080"   # interface du maître
      - "7077:7077"   # port de la grappe
      - "4040:4040"   # interface de l'application en cours
    volumes:
      - ./:/opt/project
    environment:
      - SPARK_MASTER_HOST=spark-master

  spark-worker-1:
    build: ./docker/spark
    image: bigdata-spark:3.5.1
    container_name: spark-worker-1
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
    depends_on:
      - spark-master
    volumes:
      - ./:/opt/project
    environment:
      - SPARK_WORKER_MEMORY=${SPARK_WORKER_MEMORY:-1g}
      - SPARK_WORKER_CORES=${SPARK_WORKER_CORES:-2}

  spark-worker-2:
    build: ./docker/spark
    image: bigdata-spark:3.5.1
    container_name: spark-worker-2
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
    depends_on:
      - spark-master
    volumes:
      - ./:/opt/project
    environment:
      - SPARK_WORKER_MEMORY=${SPARK_WORKER_MEMORY:-1g}
      - SPARK_WORKER_CORES=${SPARK_WORKER_CORES:-2}
```

### `docker/spark/Dockerfile`

```dockerfile
FROM apache/spark:3.5.1-python3
USER root
RUN pip install --no-cache-dir numpy
USER spark
```

### `.env.example`

```
SPARK_WORKER_MEMORY=1g
SPARK_WORKER_CORES=2
```

### `.gitignore`

```
# Données (trop volumineuses pour Git)
data/raw/*
data/bronze/*
data/silver/*
data/gold/*
data/stream/
!data/**/.gitkeep
*.csv
*.parquet
*.zip
*.log

# Résultats agrégés (petits, à versionner)
!docs/results/*.csv

# Python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# Environnement / secrets
.env

# IDE / système
.vscode/
.idea/
.DS_Store

# Spark
spark-warehouse/
metastore_db/
derby.log
```

Écrivez ce fichier avec `Out-File -Encoding ascii .gitignore` : l'encodage UTF-8 par défaut de PowerShell
ajoute un caractère invisible qui peut casser la première règle.

### `C:\Users\<vous>\.wslconfig`

```
[wsl2]
memory=5GB
processors=4
swap=4GB
```
