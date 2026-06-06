from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_spark() -> Any:
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError("pyspark is required for Databricks pipeline jobs") from exc

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    logger.info("Using active Spark session")
    return spark

