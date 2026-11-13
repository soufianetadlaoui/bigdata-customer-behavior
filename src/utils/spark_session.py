"""Création centralisée de la SparkSession (réutilisée par tous les modules)."""
from pyspark.sql import SparkSession


def get_spark(app_name: str = "customer-behavior", master: str = "local[*]") -> SparkSession:
    """master = 'local[*]' en développement, 'spark://spark-master:7077' sur le cluster Docker."""
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
