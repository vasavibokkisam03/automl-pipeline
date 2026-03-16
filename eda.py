"""
Agent 02 — Exploratory Data Analysis
Detects problem type, class balance, correlations, feature distributions.
Outputs an EDAResult that feeds the Model Selection agent.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EDAResult:
    problem_type: str = "clustering"         # "classification" | "regression" | "clustering"
    target_column: Optional[str] = None
    feature_columns: list = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)
    imbalance_ratio: Optional[float] = None
    is_imbalanced: bool = False
    correlations: list = field(default_factory=list)   # top (feat, target, corr) tuples
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    high_cardinality_cols: list = field(default_factory=list)
    insights: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    log: list = field(default_factory=list)


# Columns that strongly suggest they are the target
_TARGET_KEYWORDS = [
    "target", "label", "class", "y", "output", "churn", "survived",
    "price", "value", "salary", "income", "fraud", "default", "outcome",
    "result", "response", "dependent", "predict",
]


class EDAAgent:
    def __init__(self, target_col: Optional[str] = None, cardinality_threshold: int = 50):
        self.target_col = target_col
        self.cardinality_threshold = cardinality_threshold

    def run(self, df: pd.DataFrame) -> EDAResult:
        result = EDAResult(target_column=None, feature_columns=[])

        target = self._detect_target(df)
        result.target_column = target
        result.log.append(f"🎯 Target column detected: '{target}'" if target else "⚠️ No clear target column — defaulting to clustering")

        result.numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        result.categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        result.high_cardinality_cols = [
            c for c in result.categorical_cols
            if df[c].nunique() > self.cardinality_threshold
        ]

        result.feature_columns = [c for c in df.columns if c != target]

        result.problem_type = self._detect_problem_type(df, target)
        result.log.append(f"🔍 Problem type: {result.problem_type.upper()}")

        if result.problem_type == "classification" and target:
            result = self._class_balance(df, target, result)

        if result.numeric_cols:
            result = self._correlations(df, target, result)

        result = self._insights(df, target, result)

        return result

    # ── target detection ───────────────────────────────────────────────────────
    def _detect_target(self, df: pd.DataFrame) -> Optional[str]:
        if self.target_col and self.target_col in df.columns:
            return self.target_col
        # Last column heuristic
        last = df.columns[-1]
        # Keyword match
        for col in df.columns:
            if col.lower().strip() in _TARGET_KEYWORDS:
                return col
        # Last column if low cardinality
        if df[last].nunique() < 20:
            return last
        return None

    # ── problem type ───────────────────────────────────────────────────────────
    def _detect_problem_type(self, df: pd.DataFrame, target: Optional[str]) -> str:
        if not target:
            return "clustering"
        col = df[target]
        n_unique = col.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(col)

        if not is_numeric:
            return "classification"
        if n_unique <= 15:
            return "classification"
        if n_unique / len(df) > 0.05:
            return "regression"
        return "classification"

    # ── class balance ──────────────────────────────────────────────────────────
    def _class_balance(self, df: pd.DataFrame, target: str, result: EDAResult) -> EDAResult:
        counts = df[target].value_counts()
        result.class_counts = counts.to_dict()
        if len(counts) >= 2:
            ratio = counts.iloc[0] / counts.iloc[-1]
            result.imbalance_ratio = round(ratio, 2)
            result.is_imbalanced = ratio > 4.0
            if result.is_imbalanced:
                result.warnings.append(
                    f"⚠️ Class imbalance detected — ratio {ratio:.1f}:1. Consider SMOTE or class_weight='balanced'."
                )
                result.log.append(f"⚠️  Imbalance ratio: {ratio:.1f}:1")
        return result

    # ── correlations ───────────────────────────────────────────────────────────
    def _correlations(self, df: pd.DataFrame, target: Optional[str], result: EDAResult) -> EDAResult:
        num_df = df[result.numeric_cols].copy()
        if len(num_df.columns) < 2:
            return result

        corr_matrix = num_df.corr()

        if target and target in corr_matrix.columns:
            target_corr = (
                corr_matrix[target]
                .drop(target)
                .abs()
                .sort_values(ascending=False)
            )
            result.correlations = [
                {"feature": feat, "target": target, "value": round(float(corr_matrix[target][feat]), 3)}
                for feat in target_corr.index[:8]
            ]
            result.log.append(f"📊 Top correlated feature: '{target_corr.index[0]}' ({target_corr.iloc[0]:.3f})")
        else:
            # General top correlations
            pairs = []
            cols = corr_matrix.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append({
                        "feature": cols[i],
                        "target": cols[j],
                        "value": round(float(corr_matrix.iloc[i, j]), 3),
                    })
            pairs.sort(key=lambda x: abs(x["value"]), reverse=True)
            result.correlations = pairs[:8]

        # Flag multicollinearity
        high_pairs = [p for p in result.correlations if abs(p["value"]) > 0.9 and p["feature"] != p["target"]]
        for p in high_pairs:
            result.warnings.append(
                f"🔗 High multicollinearity: '{p['feature']}' ↔ '{p['target']}' = {p['value']}"
            )
        return result

    # ── insights ───────────────────────────────────────────────────────────────
    def _insights(self, df: pd.DataFrame, target: Optional[str], result: EDAResult) -> EDAResult:
        ins = result.insights

        ins.append(f"Dataset has {len(df):,} rows and {len(df.columns)} columns.")
        ins.append(f"{len(result.numeric_cols)} numeric, {len(result.categorical_cols)} categorical feature(s).")

        total_missing = df.isna().sum().sum()
        if total_missing > 0:
            ins.append(f"{total_missing:,} missing values remain after cleaning.")

        if result.high_cardinality_cols:
            ins.append(
                f"High-cardinality columns detected: {result.high_cardinality_cols}. Consider encoding carefully."
            )

        if result.problem_type == "classification" and result.class_counts:
            n_classes = len(result.class_counts)
            ins.append(f"Classification task with {n_classes} classes.")

        if result.correlations:
            top = result.correlations[0]
            ins.append(
                f"Strongest predictor of '{top['target']}': '{top['feature']}' (r={top['value']})."
            )

        result.insights = ins
        return result