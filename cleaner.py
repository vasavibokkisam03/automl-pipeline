"""
Agent 01 — Data Cleaning
Handles: missing values, outliers, type coercion, duplicates, sparse columns
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CleaningReport:
    original_shape: tuple
    final_shape: tuple
    duplicates_removed: int = 0
    missing_handled: dict = field(default_factory=dict)
    outliers_handled: dict = field(default_factory=dict)
    type_fixes: dict = field(default_factory=dict)
    sparse_columns_dropped: list = field(default_factory=list)
    log: list = field(default_factory=list)


class DataCleaningAgent:
    def __init__(self, missing_threshold: float = 0.6, outlier_z: float = 3.5):
        self.missing_threshold = missing_threshold  # drop col if > this % missing
        self.outlier_z = outlier_z                  # z-score threshold for outliers

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        report = CleaningReport(original_shape=df.shape, final_shape=df.shape)
        df = df.copy()

        df, report = self._fix_types(df, report)
        df, report = self._remove_duplicates(df, report)
        df, report = self._drop_sparse_columns(df, report)
        df, report = self._handle_missing(df, report)
        df, report = self._handle_outliers(df, report)

        report.final_shape = df.shape
        report.log.append(
            f"✅ Cleaning complete: {report.original_shape} → {report.final_shape}"
        )
        return df, report

    # ── type coercion ──────────────────────────────────────────────────────────
    def _fix_types(self, df: pd.DataFrame, report: CleaningReport):
        for col in df.columns:
            orig_dtype = str(df[col].dtype)

            # Try numeric coercion on object columns
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                non_null_orig = df[col].notna().sum()
                non_null_conv = converted.notna().sum()
                # Accept if we preserved ≥85% of non-null values
                if non_null_orig > 0 and non_null_conv / non_null_orig >= 0.85:
                    df[col] = converted
                    report.type_fixes[col] = f"object → numeric"
                    report.log.append(f"🔧 '{col}': coerced to numeric")
                    continue

                # Try datetime coercion
                if any(kw in col.lower() for kw in ["date", "time", "dt", "year", "month"]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        report.type_fixes[col] = f"object → datetime"
                        report.log.append(f"📅 '{col}': parsed as datetime")
                    except Exception:
                        pass

            # Downcast int64 to int32 where safe
            if df[col].dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="integer")

        return df, report

    # ── duplicates ─────────────────────────────────────────────────────────────
    def _remove_duplicates(self, df: pd.DataFrame, report: CleaningReport):
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        report.duplicates_removed = removed
        if removed:
            report.log.append(f"🗑️  Removed {removed} duplicate rows")
        return df, report

    # ── sparse columns ─────────────────────────────────────────────────────────
    def _drop_sparse_columns(self, df: pd.DataFrame, report: CleaningReport):
        sparse = [
            col for col in df.columns
            if df[col].isna().mean() > self.missing_threshold
        ]
        if sparse:
            df = df.drop(columns=sparse)
            report.sparse_columns_dropped = sparse
            report.log.append(
                f"🚫 Dropped {len(sparse)} sparse column(s): {sparse}"
            )
        return df, report

    # ── missing values ─────────────────────────────────────────────────────────
    def _handle_missing(self, df: pd.DataFrame, report: CleaningReport):
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                skewness = df[col].skew()
                if abs(skewness) > 1.0:
                    fill_val = df[col].median()
                    strategy = "median"
                else:
                    fill_val = df[col].mean()
                    strategy = "mean"
                df[col] = df[col].fillna(fill_val)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
                strategy = "forward-fill"
                fill_val = "ffill"
            else:
                fill_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
                df[col] = df[col].fillna(fill_val)
                strategy = "mode"

            report.missing_handled[col] = {
                "count": int(missing_count),
                "strategy": strategy,
                "fill_value": str(round(fill_val, 4)) if isinstance(fill_val, float) else str(fill_val),
            }
            report.log.append(
                f"🩹 '{col}': filled {missing_count} missing with {strategy} ({fill_val})"
            )

        return df, report

    # ── outliers ───────────────────────────────────────────────────────────────
    def _handle_outliers(self, df: pd.DataFrame, report: CleaningReport):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue

            z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-9))
            outlier_mask = z_scores > self.outlier_z
            outlier_count = outlier_mask.sum()

            if outlier_count == 0:
                continue

            # Cap to 1st/99th percentile instead of dropping
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)

            report.outliers_handled[col] = {
                "count": int(outlier_count),
                "strategy": "winsorize (1–99%)",
                "bounds": [round(lower, 4), round(upper, 4)],
            }
            report.log.append(
                f"📐 '{col}': capped {outlier_count} outliers to [{lower:.3f}, {upper:.3f}]"
            )

        return df, report