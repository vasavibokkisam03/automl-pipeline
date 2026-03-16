"""
Agent 03 — Model Selection
Benchmarks candidate models via cross-validation and picks the best one.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from agents.eda import EDAResult


@dataclass
class ModelBenchmark:
    name: str
    model: Any
    mean_score: float
    std_score: float
    metric: str


@dataclass
class ModelSelectionResult:
    winner_name: str
    winner_model: Any
    metric: str
    benchmarks: list = field(default_factory=list)
    feature_columns: list = field(default_factory=list)
    target_column: Optional[str] = None
    label_encoder: Optional[Any] = None
    log: list = field(default_factory=list)


class ModelSelectionAgent:
    def __init__(self, cv_folds: int = 3, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def run(self, df: pd.DataFrame, eda: EDAResult) -> ModelSelectionResult:
        result = ModelSelectionResult(
            winner_name="", winner_model=None, metric="",
            target_column=eda.target_column,
            feature_columns=eda.feature_columns,
        )

        X, y, le = self._prepare_data(df, eda)
        result.label_encoder = le

        candidates = self._get_candidates(eda.problem_type)
        metric = "f1_weighted" if eda.problem_type == "classification" else "r2"
        result.metric = metric

        result.log.append(f"🏁 Benchmarking {len(candidates)} models with {self.cv_folds}-fold CV…")

        benchmarks = []
        for name, model in candidates:
            pipe = self._build_pipeline(model, eda)
            try:
                scores = cross_val_score(pipe, X, y, cv=self.cv_folds, scoring=metric, n_jobs=-1)
                bm = ModelBenchmark(
                    name=name, model=pipe,
                    mean_score=round(float(scores.mean()), 4),
                    std_score=round(float(scores.std()), 4),
                    metric=metric,
                )
                benchmarks.append(bm)
                result.log.append(f"  {name:30s} {metric}={bm.mean_score:.4f} ± {bm.std_score:.4f}")
            except Exception as e:
                result.log.append(f"  ⚠️ {name} failed: {e}")

        if not benchmarks:
            raise RuntimeError("All models failed during benchmarking.")

        benchmarks.sort(key=lambda b: b.mean_score, reverse=True)
        result.benchmarks = benchmarks
        winner = benchmarks[0]
        result.winner_name = winner.name
        result.winner_model = winner.model
        result.log.append(f"🏆 Winner: {winner.name} ({metric}={winner.mean_score:.4f})")

        return result

    # ── data prep ──────────────────────────────────────────────────────────────
    def _prepare_data(self, df: pd.DataFrame, eda: EDAResult):
        target = eda.target_column
        features = [c for c in eda.feature_columns if c in df.columns]

        X = df[features].copy()
        y = df[target].copy() if target else None

        # Encode categoricals in X
        for col in X.select_dtypes(include=["object", "category", "datetime64[ns]"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        le = None
        if y is not None and (y.dtype == object or str(y.dtype) == "category"):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

        return X, y, le

    # ── pipeline ───────────────────────────────────────────────────────────────
    def _build_pipeline(self, model, eda: EDAResult) -> Pipeline:
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
        return Pipeline(steps)

    # ── candidate models ───────────────────────────────────────────────────────
    def _get_candidates(self, problem_type: str) -> list:
        rs = self.random_state
        if problem_type == "classification":
            candidates = [
                ("Logistic Regression",    LogisticRegression(max_iter=500, random_state=rs)),
                ("Decision Tree",          DecisionTreeClassifier(random_state=rs)),
                ("Random Forest",          RandomForestClassifier(n_estimators=100, random_state=rs)),
                ("Gradient Boosting",      GradientBoostingClassifier(n_estimators=100, random_state=rs)),
            ]
            if HAS_XGB:
                candidates.append(
                    ("XGBoost", XGBClassifier(n_estimators=100, random_state=rs, verbosity=0, eval_metric="logloss"))
                )
        else:
            candidates = [
                ("Linear Regression",      LinearRegression()),
                ("Ridge Regression",       Ridge(random_state=rs)),
                ("Random Forest",          RandomForestRegressor(n_estimators=100, random_state=rs)),
                ("Gradient Boosting",      GradientBoostingRegressor(n_estimators=100, random_state=rs)),
            ]
            if HAS_XGB:
                candidates.append(
                    ("XGBoost", XGBRegressor(n_estimators=100, random_state=rs, verbosity=0))
                )
        return candidates