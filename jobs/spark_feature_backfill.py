"""Databricks / Apache Spark distributed point-in-time feature backfill job.

Executes the exact same `OutGuardFeatureExtractor` used by online serving across
tenant-partitioned, chronologically sorted Delta Lake / Parquet outbound email logs
via PySpark `groupBy("tenant_id").applyInPandas(...)`. This eliminates online/offline
training-serving skew while scaling horizontally across thousands of enterprise tenants
on AWS EMR or Databricks.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from outguard.config import PipelineConfig
from outguard.features.extractor import OutGuardFeatureExtractor
from outguard.schemas import ALL_FEATURES


def backfill_tenant_partition(
    pdf_iter: Iterator[pd.DataFrame],
    cfg_dict: dict,
    brand_map: dict[str, str],
) -> Iterator[pd.DataFrame]:
    """Stateful chronological mapper executed per-tenant inside Spark workers."""
    cfg = PipelineConfig.model_validate(cfg_dict)
    extractor = OutGuardFeatureExtractor(cfg, domain_to_brand=brand_map)
    state = extractor.create_empty_state()

    for pdf in pdf_iter:
        ordered = pdf.sort_values("timestamp_epoch", kind="mergesort").reset_index(drop=True)
        feat_rows = [
            extractor.extract_single_event(rec, state, mutate_state=True)
            for rec in ordered.to_dict(orient="records")
        ]
        feat_df = pd.DataFrame(feat_rows, columns=ALL_FEATURES)
        yield pd.concat([ordered[["event_id", "tenant_id", "timestamp_epoch", "split"]], feat_df], axis=1)


def build_spark_backfill_plan(
    spark_session,
    input_delta_table: str,
    output_delta_table: str,
    config_path: str = "configs/default_pipeline.yaml",
) -> None:
    """Entrypoint for scheduled Databricks / AWS EMR offline training table generation."""
    cfg = PipelineConfig.from_yaml(config_path)
    cfg_dict = cfg.model_dump()

    events_sdf = spark_session.read.table(input_delta_table)

    def _udf_runner(pdf: pd.DataFrame) -> pd.DataFrame:
        out_chunks = list(backfill_tenant_partition(iter([pdf]), cfg_dict, {}))
        return out_chunks[0]

    # Schema matches event metadata + 34 canonical float64 features
    feat_DDL = ", ".join(f"{col} DOUBLE" for col in ALL_FEATURES)
    output_schema = f"event_id STRING, tenant_id STRING, timestamp_epoch DOUBLE, split STRING, {feat_DDL}"

    features_sdf = events_sdf.groupBy("tenant_id").applyInPandas(
        _udf_runner, schema=output_schema
    )
    (
        features_sdf.write.format("delta")
        .mode("overwrite")
        .partitionBy("split")
        .saveAsTable(output_delta_table)
    )
