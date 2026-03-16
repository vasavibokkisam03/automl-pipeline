"""
Agent 04 — Training & Evaluation
Proper train/test split, hyperparameter tuning, full metrics + plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Any, Optional

from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error,
)
from sklearn.impute import SimpleImputer

from agents.eda import EDAResult
from agents.model_selector import ModelSelectionResult


@dataclass
class TrainingResult:
    problem_type: str
    model_name: str
    trained_model: Any
    metrics: dict = field(default_factory=dict)
    classification_report: Optional[str] = None
    confusion_matrix: Optional[Any] = None
    feature_importances: Optional[pd.Series] = None
    figures: dict = field(default_factory=dict)   # name → matplotlib Figure
    log: list = field(default_factory=list)


class TrainingAgent:
    def __init__(self, test_size: float = 0.2, random_state: int = 42, tune: bool = True):
        self.test_size = test_size
        self.random_state = random_state
        self.tune = tune

    def run(
        self,
        df: pd.DataFrame,
        eda: EDAResult,
        selection: ModelSelectionResult,
    ) -> TrainingResult:
        result = TrainingResult(
            problem_type=eda.problem_type,
            model_name=selection.winner_name,
            trained_model=None,
        )

        X, y, le = self._prepare(df, eda, selection)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if eda.problem_type == "classification" else None,
        )
        result.log.append(f"📂 Train: {len(X_train)} | Test: {len(X_test)}")

        model = selection.winner_model
        if self.tune:
            model, result = self._tune(model, X_train, y_train, eda.problem_type, result)

        model.fit(X_train, y_train)
        result.trained_model = model
        result.log.append(f"✅ Model trained: {selection.winner_name}")

        y_pred = model.predict(X_test)
        result = self._evaluate(y_test, y_pred, eda.problem_type, result)
        result = self._feature_importance(model, eda.feature_columns, result)
        result = self._make_figures(model, X_train, y_train, X_test, y_test, y_pred, eda, result)

        return result

    # ── data prep ──────────────────────────────────────────────────────────────
    def _prepare(self, df, eda, selection):
        features = [c for c in eda.feature_columns if c in df.columns]
        X = df[features].copy()
        y = df[eda.target_column].copy()

        for col in X.select_dtypes(include=["object", "category", "datetime64[ns]"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Fill remaining NaN
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        le = None
        if y.dtype == object or str(y.dtype) == "category":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)))

        return X, y, le

    # ── hyperparameter tuning ──────────────────────────────────────────────────
    def _tune(self, model, X_train, y_train, problem_type, result):
        name = result.model_name
        param_grid = self._param_grid(name)
        if not param_grid:
            result.log.append("⚙️  No tuning config for this model — skipping.")
            return model, result

        search = RandomizedSearchCV(
            model, param_grid, n_iter=10, cv=3,
            scoring="f1_weighted" if problem_type == "classification" else "r2",
            random_state=self.random_state, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        result.log.append(f"🔧 Best params: {search.best_params_}")
        return search.best_estimator_, result

    def _param_grid(self, name: str) -> dict:
        grids = {
            "Random Forest": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
            },
            "XGBoost": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
            },
            "Logistic Regression": {
                "model__C": [0.01, 0.1, 1, 10, 100],
            },
            "Ridge Regression": {
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
        }
        for key in grids:
            if key in name:
                return grids[key]
        return {}

    # ── evaluation ─────────────────────────────────────────────────────────────
    def _evaluate(self, y_test, y_pred, problem_type, result):
        if problem_type == "classification":
            result.metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            result.metrics["f1_weighted"] = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
            result.classification_report = classification_report(y_test, y_pred, zero_division=0)
            result.confusion_matrix = confusion_matrix(y_test, y_pred)
            result.log.append(f"📈 Accuracy: {result.metrics['accuracy']} | F1: {result.metrics['f1_weighted']}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            result.metrics["rmse"] = round(float(np.sqrt(mse)), 4)
            result.metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            result.metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)
            result.log.append(f"📈 RMSE: {result.metrics['rmse']} | R²: {result.metrics['r2']}")
        return result

    # ── feature importance ─────────────────────────────────────────────────────
    def _feature_importance(self, model, feature_cols, result):
        try:
            step = model.named_steps.get("model") if hasattr(model, "named_steps") else model
            if hasattr(step, "feature_importances_"):
                fi = pd.Series(step.feature_importances_, index=feature_cols[:len(step.feature_importances_)])
                result.feature_importances = fi.sort_values(ascending=False)
            elif hasattr(step, "coef_"):
                coef = step.coef_.ravel() if step.coef_.ndim > 1 else step.coef_
                fi = pd.Series(np.abs(coef), index=feature_cols[:len(coef)])
                result.feature_importances = fi.sort_values(ascending=False)
        except Exception:
            pass
        return result

    # ── figures ────────────────────────────────────────────────────────────────
    def _make_figures(self, model, X_train, y_train, X_test, y_test, y_pred, eda, result):
        DARK = "#0A0C10"
        PANEL = "#0D1117"
        ACCENT = "#00E5A0"
        TEXT = "#C8D6E5"
        GRID = "#1C2130"

        def style_ax(ax):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT, labelsize=9)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            ax.title.set_color(TEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)

        # ── 1. Confusion matrix / actual vs predicted ─────────────────────────
        fig1, ax = plt.subplots(figsize=(6, 5), facecolor=DARK)
        style_ax(ax)
        if eda.problem_type == "classification" and result.confusion_matrix is not None:
            cm = result.confusion_matrix
            im = ax.imshow(cm, cmap="Greens")
            ax.set_title("Confusion Matrix", pad=12)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=TEXT, fontsize=10)
            plt.colorbar(im, ax=ax)
        else:
            lim = min(200, len(y_test))
            ax.scatter(range(lim), y_test[:lim], color=ACCENT, s=15, alpha=0.7, label="Actual")
            ax.scatter(range(lim), y_pred[:lim], color="#FF6B6B", s=15, alpha=0.7, label="Predicted")
            ax.set_title("Actual vs Predicted (first 200 samples)")
            ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        fig1.tight_layout()
        result.figures["prediction"] = fig1

        # ── 2. Feature importance ─────────────────────────────────────────────
        if result.feature_importances is not None:
            top_n = result.feature_importances.head(15)
            fig2, ax = plt.subplots(figsize=(7, max(4, len(top_n) * 0.4)), facecolor=DARK)
            style_ax(ax)
            colors = [ACCENT if i == 0 else "#2A5A4A" for i in range(len(top_n))]
            ax.barh(top_n.index[::-1], top_n.values[::-1], color=colors[::-1], edgecolor=GRID)
            ax.set_title("Feature Importance (Top 15)")
            ax.set_xlabel("Importance Score")
            fig2.tight_layout()
            result.figures["feature_importance"] = fig2

        # ── 3. Learning curve ─────────────────────────────────────────────────
        try:
            sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                cv=3, n_jobs=-1,
                train_sizes=np.linspace(0.2, 1.0, 6),
                scoring="f1_weighted" if eda.problem_type == "classification" else "r2",
            )
            fig3, ax = plt.subplots(figsize=(7, 4), facecolor=DARK)
            style_ax(ax)
            ax.plot(sizes, train_scores.mean(axis=1), color=ACCENT, label="Train", linewidth=2)
            ax.fill_between(sizes,
                            train_scores.mean(axis=1) - train_scores.std(axis=1),
                            train_scores.mean(axis=1) + train_scores.std(axis=1),
                            alpha=0.15, color=ACCENT)
            ax.plot(sizes, val_scores.mean(axis=1), color="#FF6B6B", label="Validation", linewidth=2)
            ax.fill_between(sizes,
                            val_scores.mean(axis=1) - val_scores.std(axis=1),
                            val_scores.mean(axis=1) + val_scores.std(axis=1),
                            alpha=0.15, color="#FF6B6B")
            ax.set_title("Learning Curve")
            ax.set_xlabel("Training Samples")
            ax.set_ylabel("Score")
            ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
            ax.grid(True, color=GRID, alpha=0.5)
            fig3.tight_layout()
            result.figures["learning_curve"] = fig3
        except Exception:
            pass

        return result