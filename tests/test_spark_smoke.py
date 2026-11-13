from src.utils.spark_session import get_spark


def test_spark_runs():
    spark = get_spark("smoke-test")
    assert spark.range(1000).count() == 1000
    spark.stop()
