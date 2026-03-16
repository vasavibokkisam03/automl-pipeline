"""
Agent 05 — Explainability
SHAP values + plain-English insight generation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any, Optional

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from agents.eda import EDAResult
from agents.trainer import TrainingResult

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class ExplainabilityResult:
    shap_available: bool
    top_features: list = field(default_factory=list)   # [(feature, mean_abs_shap)]
    plain_english: list = field(default_factory=list)  # human-readable sentences
    figures: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


class ExplainabilityAgent:
    def run(
        self,
        df: pd.DataFrame,
        eda: EDAResult,
        training: TrainingResult,
    ) -> ExplainabilityResult:
        result = ExplainabilityResult(shap_available=HAS_SHAP)

        if not HAS_SHAP:
            result.log.append("⚠️ SHAP not installed. Using feature importances instead.")
            result = self._fallback_importance(training, result)
            result = self._generate_plain_english(training, eda, result)
            return result

        X = self._prepare(df, eda)

        try:
            result = self._shap_analysis(X, training, eda, result)
        except Exception as e:
            result.log.append(f"⚠️ SHAP failed ({e}). Falling back to feature importances.")
            result = self._fallback_importance(training, result)

        result = self._generate_plain_english(training, eda, result)
        return result

    # ── data prep ──────────────────────────────────────────────────────────────
    def _prepare(self, df, eda):
        features = [c for c in eda.feature_columns if c in df.columns]
        X = df[features].copy()
        for col in X.select_dtypes(include=["object", "category", "datetime64[ns]"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        return X

    # ── shap ───────────────────────────────────────────────────────────────────
    def _shap_analysis(self, X, training: TrainingResult, eda: EDAResult, result: ExplainabilityResult):
        model = training.trained_model
        sample = X.sample(min(200, len(X)), random_state=42)

        # Get the inner model from pipeline
        if hasattr(model, "named_steps"):
            inner = model.named_steps["model"]
            # Transform sample through pre-processing steps
            X_transformed = sample.copy()
            for name, step in model.named_steps.items():
                if name == "model":
                    break
                X_transformed = pd.DataFrame(step.transform(X_transformed), columns=sample.columns)
        else:
            inner = model
            X_transformed = sample

        # Choose explainer
        model_type = type(inner).__name__
        if any(t in model_type for t in ["XGB", "GradientBoosting", "RandomForest", "DecisionTree"]):
            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(X_transformed)
        else:
            explainer = shap.LinearExplainer(inner, X_transformed)
            shap_values = explainer.shap_values(X_transformed)

        # Handle multi-class
        if isinstance(shap_values, list):
            shap_arr = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_arr = np.abs(shap_values)

        mean_shap = pd.Series(shap_arr.mean(axis=0), index=sample.columns)
        mean_shap = mean_shap.sort_values(ascending=False)
        result.top_features = [(f, round(float(v), 4)) for f, v in mean_shap.head(15).items()]
        result.log.append(f"🔍 Top SHAP feature: '{result.top_features[0][0]}'")

        # SHAP bar plot
        DARK, PANEL, ACCENT, TEXT, GRID = "#0A0C10", "#0D1117", "#00E5A0", "#C8D6E5", "#1C2130"
        top = mean_shap.head(12)
        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.45)), facecolor=DARK)
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        colors = [ACCENT if i < 3 else "#2A5A4A" for i in range(len(top))]
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor=GRID)
        ax.set_title("SHAP Feature Importance", color=TEXT, pad=10)
        ax.set_xlabel("Mean |SHAP value|", color=TEXT)
        fig.tight_layout()
        result.figures["shap_bar"] = fig

        return result

    # ── fallback ───────────────────────────────────────────────────────────────
    def _fallback_importance(self, training: TrainingResult, result: ExplainabilityResult):
        if training.feature_importances is not None:
            fi = training.feature_importances
            result.top_features = [(f, round(float(v), 4)) for f, v in fi.head(15).items()]
        return result

    # ── plain English ──────────────────────────────────────────────────────────
    def _generate_plain_english(
        self, training: TrainingResult, eda: EDAResult, result: ExplainabilityResult
    ) -> ExplainabilityResult:
        lines = []
        m = training.metrics
        pt = training.problem_type

        # Performance summary
        if pt == "classification":
            acc = m.get("accuracy", 0)
            f1 = m.get("f1_weighted", 0)
            perf_label = "excellent" if acc >= 0.90 else "good" if acc >= 0.75 else "moderate"
            lines.append(
                f"The model achieves **{acc*100:.1f}% accuracy** with a weighted F1 score of "
                f"**{f1:.3f}** — {perf_label} performance overall."
            )
        else:
            r2 = m.get("r2", 0)
            rmse = m.get("rmse", 0)
            pct = max(0, r2 * 100)
            lines.append(
                f"The model explains **{pct:.1f}% of variance** in the target (R² = {r2:.3f}), "
                f"with an average prediction error of **{rmse:.3f}** (RMSE)."
            )

        # Top features
        if result.top_features:
            top3 = [f for f, _ in result.top_features[:3]]
            if len(top3) == 1:
                feat_str = f"**{top3[0]}**"
            elif len(top3) == 2:
                feat_str = f"**{top3[0]}** and **{top3[1]}**"
            else:
                feat_str = f"**{top3[0]}**, **{top3[1]}**, and **{top3[2]}**"
            lines.append(
                f"The most influential features were {feat_str}, which together drove "
                f"the majority of the model's predictions."
            )

        # Target info
        if eda.target_column:
            lines.append(
                f"The model was trained to predict **'{eda.target_column}'** using "
                f"{len(eda.feature_columns)} input features."
            )

        # Imbalance note
        if eda.is_imbalanced:
            lines.append(
                f"⚠️ The training data had class imbalance (ratio {eda.imbalance_ratio}:1). "
                f"Results on minority classes may be less reliable."
            )

        # Dataset size note
        if pt == "classification" and m.get("accuracy", 0) > 0.95:
            lines.append(
                "🔎 Accuracy above 95% — double-check for data leakage or overfitting."
            )

        result.plain_english = lines
        return result