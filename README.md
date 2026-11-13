# Analyse du comportement client à grande échelle

Projet de fin de module **Big Data** : plateforme distribuée pour ingérer, traiter et analyser des dizaines de millions d'événements e-commerce (segmentation, prédiction du churn, recommandation, streaming).

## Objectifs
- Pipeline Bronze → Silver → Gold avec Spark et Parquet
- Entonnoir de conversion, cohortes, segmentation RFM + K-Means
- Prédiction du churn et recommandation (ALS)
- Streaming simulé (Kafka + Spark Structured Streaming)
- Benchmarks de scalabilité et tableau de bord

## Architecture
```
Données brutes (CSV) → Spark → Data Lake MinIO/HDFS (Bronze/Silver/Gold)
                                   → Spark MLlib → Dashboard
```

## Stack
Apache Spark 3.5 · Parquet sur système de fichiers partagé · Parquet · Kafka (plus tard) · Airflow (plus tard) · Streamlit · Docker

## Démarrage rapide
```bash
git clone <URL_DU_DEPOT>
cd bigdata-customer-behavior
cp .env.example .env
docker compose up -d
```
- Spark Master UI : http://localhost:8080

## Données
Dataset : *eCommerce behavior data from multi-category store* (Kaggle, REES46).
Placer les fichiers CSV dans `data/raw/` (non versionnés).

## Structure
| Dossier | Rôle |
|---------|------|
| `data/` | Données locales (ignorées par Git) |
| `src/ingestion` | Chargement CSV → Bronze |
| `src/processing` | Nettoyage et features (Silver/Gold) |
| `src/ml` | Segmentation, churn, recommandation |
| `src/streaming` | Kafka et Structured Streaming |
| `dags/` | Orchestration Airflow |
| `dashboard/` | Application de visualisation |
| `notebooks/` | Analyses exploratoires |
| `benchmarks/` | Mesures de performance |
| `docs/` | Rapport et schémas |

## Avancement
- [x] Étape 1 : Initialisation du dépôt et de l'infrastructure
- [ ] Étape 2 : Ingestion et conversion en Parquet
- [ ] Étape 3 : Nettoyage et couche Silver
- [ ] Étape 4 : Analyse exploratoire et entonnoir
- [ ] Étape 5 : Segmentation RFM + clustering
- [ ] Étape 6 : Churn et recommandation
- [ ] Étape 7 : Streaming, dashboard, Airflow
- [ ] Étape 8 : Benchmarks et rapport

## Auteurs
À compléter.
