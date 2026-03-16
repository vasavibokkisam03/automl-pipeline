"""
AutoML Pipeline — Single File
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 01 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════

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


def run_cleaning(df: pd.DataFrame) -> tuple:
    report = CleaningReport(original_shape=df.shape, final_shape=df.shape)
    df = df.copy()

    # Fix types
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null_orig = df[col].notna().sum()
            non_null_conv = converted.notna().sum()
            if non_null_orig > 0 and non_null_conv / non_null_orig >= 0.85:
                df[col] = converted
                report.type_fixes[col] = "object → numeric"
                report.log.append(f"🔧 '{col}': coerced to numeric")
            elif any(kw in col.lower() for kw in ["date", "time", "dt", "year", "month"]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    report.type_fixes[col] = "object → datetime"
                    report.log.append(f"📅 '{col}': parsed as datetime")
                except Exception:
                    pass

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)
    if report.duplicates_removed:
        report.log.append(f"🗑️ Removed {report.duplicates_removed} duplicate rows")

    # Drop sparse columns (>60% missing)
    sparse = [c for c in df.columns if df[c].isna().mean() > 0.6]
    if sparse:
        df = df.drop(columns=sparse)
        report.sparse_columns_dropped = sparse
        report.log.append(f"🚫 Dropped sparse columns: {sparse}")

    # Handle missing values
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
            df[col] = df[col].ffill().bfill()
            strategy, fill_val = "forward-fill", "ffill"
        else:
            fill_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
            df[col] = df[col].fillna(fill_val)
            strategy = "mode"
        report.missing_handled[col] = {
            "count": int(missing_count), "strategy": strategy,
            "fill_value": str(round(fill_val, 4)) if isinstance(fill_val, float) else str(fill_val),
        }
        report.log.append(f"🩹 '{col}': filled {missing_count} missing → {strategy}")

    # Handle outliers (winsorize)
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue
        z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-9))
        outlier_count = (z_scores > 3.5).sum()
        if outlier_count == 0:
            continue
        lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)
        report.outliers_handled[col] = {
            "count": int(outlier_count), "strategy": "winsorize (1–99%)",
            "bounds": [round(lower, 4), round(upper, 4)],
        }
        report.log.append(f"📐 '{col}': capped {outlier_count} outliers")

    report.final_shape = df.shape
    report.log.append(f"✅ Cleaning complete: {report.original_shape} → {report.final_shape}")
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 02 — EDA
# ══════════════════════════════════════════════════════════════════════════════

_TARGET_KEYWORDS = [
    "target", "label", "class", "y", "output", "churn", "survived",
    "price", "value", "salary", "income", "fraud", "default", "outcome",
    "result", "response", "dependent", "predict",
]

@dataclass
class EDAResult:
    problem_type: str
    target_column: Optional[str]
    feature_columns: list
    class_counts: dict = field(default_factory=dict)
    imbalance_ratio: Optional[float] = None
    is_imbalanced: bool = False
    correlations: list = field(default_factory=list)
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    high_cardinality_cols: list = field(default_factory=list)
    insights: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    log: list = field(default_factory=list)


def run_eda(df: pd.DataFrame, target_col: Optional[str] = None) -> EDAResult:
    # Detect target
    target = None
    if target_col and target_col in df.columns:
        target = target_col
    else:
        for col in df.columns:
            if col.lower().strip() in _TARGET_KEYWORDS:
                target = col
                break
        if not target:
            last = df.columns[-1]
            if df[last].nunique() < 20:
                target = last

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    high_cardinality_cols = [c for c in categorical_cols if df[c].nunique() > 50]
    feature_columns = [c for c in df.columns if c != target]

    # Problem type
    problem_type = "clustering"
    if target:
        col = df[target]
        n_unique = col.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(col)
        if not is_numeric or n_unique <= 15:
            problem_type = "classification"
        elif n_unique / len(df) > 0.05:
            problem_type = "regression"
        else:
            problem_type = "classification"

    result = EDAResult(
        problem_type=problem_type, target_column=target,
        feature_columns=feature_columns, numeric_cols=numeric_cols,
        categorical_cols=categorical_cols, high_cardinality_cols=high_cardinality_cols,
    )
    result.log.append(f"🎯 Target: '{target}' | Problem: {problem_type.upper()}")

    # Class balance
    if problem_type == "classification" and target:
        counts = df[target].value_counts()
        result.class_counts = counts.to_dict()
        if len(counts) >= 2:
            ratio = counts.iloc[0] / counts.iloc[-1]
            result.imbalance_ratio = round(ratio, 2)
            result.is_imbalanced = ratio > 4.0
            if result.is_imbalanced:
                result.warnings.append(f"⚠️ Class imbalance: ratio {ratio:.1f}:1. Consider class_weight='balanced'.")

    # Correlations
    if len(numeric_cols) >= 2:
        num_df = df[numeric_cols].copy()
        corr_matrix = num_df.corr()
        if target and target in corr_matrix.columns:
            target_corr = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
            result.correlations = [
                {"feature": f, "target": target, "value": round(float(corr_matrix[target][f]), 3)}
                for f in target_corr.index[:8]
            ]
        else:
            pairs = []
            cols = corr_matrix.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append({"feature": cols[i], "target": cols[j], "value": round(float(corr_matrix.iloc[i, j]), 3)})
            pairs.sort(key=lambda x: abs(x["value"]), reverse=True)
            result.correlations = pairs[:8]

        # Multicollinearity warnings
        for p in result.correlations:
            if abs(p["value"]) > 0.9 and p["feature"] != p["target"]:
                result.warnings.append(f"🔗 High multicollinearity: '{p['feature']}' ↔ '{p['target']}' = {p['value']}")

    # Insights
    result.insights = [
        f"Dataset has {len(df):,} rows and {len(df.columns)} columns.",
        f"{len(numeric_cols)} numeric and {len(categorical_cols)} categorical feature(s).",
    ]
    if df.isna().sum().sum() > 0:
        result.insights.append(f"{df.isna().sum().sum():,} missing values remain after cleaning.")
    if high_cardinality_cols:
        result.insights.append(f"High-cardinality columns: {high_cardinality_cols}. Encode carefully.")
    if result.correlations:
        top = result.correlations[0]
        result.insights.append(f"Strongest predictor of '{top['target']}': '{top['feature']}' (r={top['value']}).")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 03 — MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════

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


def _prepare_xy(df, eda):
    features = [c for c in eda.feature_columns if c in df.columns]
    X = df[features].copy()
    y = df[eda.target_column].copy() if eda.target_column else None
    for col in X.select_dtypes(include=["object", "category", "datetime64[ns]"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    le = None
    if y is not None and (y.dtype == object or str(y.dtype) == "category"):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    return X, y, le


def _build_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def run_model_selection(df: pd.DataFrame, eda: EDAResult) -> ModelSelectionResult:
    rs = 42
    result = ModelSelectionResult(
        winner_name="", winner_model=None, metric="",
        target_column=eda.target_column, feature_columns=eda.feature_columns,
    )

    X, y, le = _prepare_xy(df, eda)
    result.label_encoder = le
    metric = "f1_weighted" if eda.problem_type == "classification" else "r2"
    result.metric = metric

    if eda.problem_type == "classification":
        candidates = [
            ("Logistic Regression",   LogisticRegression(max_iter=500, random_state=rs)),
            ("Decision Tree",         DecisionTreeClassifier(random_state=rs)),
            ("Random Forest",         RandomForestClassifier(n_estimators=100, random_state=rs)),
            ("Gradient Boosting",     GradientBoostingClassifier(n_estimators=100, random_state=rs)),
        ]
        if HAS_XGB:
            candidates.append(("XGBoost", XGBClassifier(n_estimators=100, random_state=rs, verbosity=0, eval_metric="logloss")))
    else:
        candidates = [
            ("Linear Regression",     LinearRegression()),
            ("Ridge Regression",      Ridge(random_state=rs)),
            ("Random Forest",         RandomForestRegressor(n_estimators=100, random_state=rs)),
            ("Gradient Boosting",     GradientBoostingRegressor(n_estimators=100, random_state=rs)),
        ]
        if HAS_XGB:
            candidates.append(("XGBoost", XGBRegressor(n_estimators=100, random_state=rs, verbosity=0)))

    result.log.append(f"🏁 Benchmarking {len(candidates)} models (3-fold CV)…")
    benchmarks = []
    for name, model in candidates:
        pipe = _build_pipeline(model)
        try:
            scores = cross_val_score(pipe, X, y, cv=3, scoring=metric, n_jobs=-1)
            bm = ModelBenchmark(name=name, model=pipe,
                                mean_score=round(float(scores.mean()), 4),
                                std_score=round(float(scores.std()), 4),
                                metric=metric)
            benchmarks.append(bm)
            result.log.append(f"  {name:28s} {metric}={bm.mean_score:.4f} ± {bm.std_score:.4f}")
        except Exception as e:
            result.log.append(f"  ⚠️ {name} failed: {e}")

    if not benchmarks:
        raise RuntimeError("All models failed during benchmarking.")

    benchmarks.sort(key=lambda b: b.mean_score, reverse=True)
    result.benchmarks = benchmarks
    result.winner_name = benchmarks[0].name
    result.winner_model = benchmarks[0].model
    result.log.append(f"🏆 Winner: {benchmarks[0].name} ({metric}={benchmarks[0].mean_score:.4f})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 04 — TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    problem_type: str
    model_name: str
    trained_model: Any
    metrics: dict = field(default_factory=dict)
    classification_report_str: Optional[str] = None
    confusion_matrix: Optional[Any] = None
    feature_importances: Optional[pd.Series] = None
    figures: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def run_training(df: pd.DataFrame, eda: EDAResult, selection: ModelSelectionResult) -> TrainingResult:
    result = TrainingResult(problem_type=eda.problem_type, model_name=selection.winner_name, trained_model=None)
    rs = 42

    features = [c for c in eda.feature_columns if c in df.columns]
    X = df[features].copy()
    y = df[eda.target_column].copy()

    for col in X.select_dtypes(include=["object", "category", "datetime64[ns]"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    le = None
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)))

    stratify = y if eda.problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=stratify)
    result.log.append(f"📂 Train: {len(X_train)} | Test: {len(X_test)}")

    # Quick hyperparameter tuning
    model = selection.winner_model
    param_grids = {
        "Random Forest":      {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5]},
        "Gradient Boosting":  {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1, 0.2], "model__max_depth": [3, 5]},
        "XGBoost":            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5]},
        "Logistic Regression": {"model__C": [0.1, 1, 10]},
        "Ridge Regression":   {"model__alpha": [0.1, 1.0, 10.0]},
    }
    pg = next((v for k, v in param_grids.items() if k in selection.winner_name), None)
    if pg:
        try:
            search = RandomizedSearchCV(model, pg, n_iter=8, cv=3,
                                        scoring="f1_weighted" if eda.problem_type == "classification" else "r2",
                                        random_state=rs, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            result.log.append(f"🔧 Best params: {search.best_params_}")
        except Exception as e:
            result.log.append(f"⚠️ Tuning failed ({e}), using default params")

    model.fit(X_train, y_train)
    result.trained_model = model
    y_pred = model.predict(X_test)

    # Metrics
    if eda.problem_type == "classification":
        result.metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        result.metrics["f1_weighted"] = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        result.classification_report_str = classification_report(y_test, y_pred, zero_division=0)
        result.confusion_matrix = confusion_matrix(y_test, y_pred)
        result.log.append(f"📈 Accuracy: {result.metrics['accuracy']} | F1: {result.metrics['f1_weighted']}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        result.metrics["rmse"] = round(float(np.sqrt(mse)), 4)
        result.metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
        result.metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)
        result.log.append(f"📈 RMSE: {result.metrics['rmse']} | R²: {result.metrics['r2']}")

    # Feature importance
    try:
        step = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        if hasattr(step, "feature_importances_"):
            fi = pd.Series(step.feature_importances_, index=features[:len(step.feature_importances_)])
            result.feature_importances = fi.sort_values(ascending=False)
        elif hasattr(step, "coef_"):
            coef = step.coef_.ravel() if step.coef_.ndim > 1 else step.coef_
            fi = pd.Series(np.abs(coef), index=features[:len(coef)])
            result.feature_importances = fi.sort_values(ascending=False)
    except Exception:
        pass

    # Plots
    DARK, PANEL, ACCENT, TEXT, GRID = "#0A0C10", "#0D1117", "#00E5A0", "#C8D6E5", "#1C2130"

    def style_ax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # Confusion matrix / actual vs predicted
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
        ax.scatter(range(lim), list(y_test)[:lim], color=ACCENT, s=15, alpha=0.7, label="Actual")
        ax.scatter(range(lim), y_pred[:lim], color="#FF6B6B", s=15, alpha=0.7, label="Predicted")
        ax.set_title("Actual vs Predicted (first 200 samples)")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    fig1.tight_layout()
    result.figures["prediction"] = fig1

    # Feature importance plot
    if result.feature_importances is not None:
        top_n = result.feature_importances.head(15)
        fig2, ax = plt.subplots(figsize=(7, max(4, len(top_n) * 0.4)), facecolor=DARK)
        style_ax(ax)
        colors = [ACCENT if i == 0 else "#1A4A3A" for i in range(len(top_n))]
        ax.barh(top_n.index[::-1], top_n.values[::-1], color=colors[::-1], edgecolor=GRID)
        ax.set_title("Feature Importance (Top 15)")
        ax.set_xlabel("Importance Score")
        fig2.tight_layout()
        result.figures["feature_importance"] = fig2

    # Learning curve
    try:
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 6),
            scoring="f1_weighted" if eda.problem_type == "classification" else "r2",
        )
        fig3, ax = plt.subplots(figsize=(7, 4), facecolor=DARK)
        style_ax(ax)
        ax.plot(sizes, train_scores.mean(axis=1), color=ACCENT, label="Train", linewidth=2)
        ax.fill_between(sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color=ACCENT)
        ax.plot(sizes, val_scores.mean(axis=1), color="#FF6B6B", label="Validation", linewidth=2)
        ax.fill_between(sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color="#FF6B6B")
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


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 05 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExplainabilityResult:
    shap_available: bool
    top_features: list = field(default_factory=list)
    plain_english: list = field(default_factory=list)
    figures: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def run_explainability(df: pd.DataFrame, eda: EDAResult, training: TrainingResult) -> ExplainabilityResult:
    result = ExplainabilityResult(shap_available=HAS_SHAP)
    DARK, PANEL, ACCENT, TEXT, GRID = "#0A0C10", "#0D1117", "#00E5A0", "#C8D6E5", "#1C2130"

    features = [c for c in eda.feature_columns if c in df.columns]
    X = df[features].copy()
    for col in X.select_dtypes(include=["object", "category", "datetime64[ns]"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # SHAP
    if HAS_SHAP:
        try:
            model = training.trained_model
            sample = X.sample(min(200, len(X)), random_state=42)
            if hasattr(model, "named_steps"):
                inner = model.named_steps["model"]
                X_transformed = sample.copy()
                for name, step in model.named_steps.items():
                    if name == "model":
                        break
                    X_transformed = pd.DataFrame(step.transform(X_transformed), columns=sample.columns)
            else:
                inner, X_transformed = model, sample

            model_type = type(inner).__name__
            if any(t in model_type for t in ["XGB", "GradientBoosting", "RandomForest", "DecisionTree"]):
                explainer = shap.TreeExplainer(inner)
                shap_values = explainer.shap_values(X_transformed)
            else:
                explainer = shap.LinearExplainer(inner, X_transformed)
                shap_values = explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):
                shap_arr = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_arr = np.abs(shap_values)

            mean_shap = pd.Series(shap_arr.mean(axis=0), index=sample.columns).sort_values(ascending=False)
            result.top_features = [(f, round(float(v), 4)) for f, v in mean_shap.head(15).items()]
            result.log.append(f"🔍 Top SHAP feature: '{result.top_features[0][0]}'")

            # SHAP bar plot
            top = mean_shap.head(12)
            fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.45)), facecolor=DARK)
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT, labelsize=9)
            for spine in ax.spines.values(): spine.set_edgecolor(GRID)
            colors = [ACCENT if i < 3 else "#1A4A3A" for i in range(len(top))]
            ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor=GRID)
            ax.set_title("SHAP Feature Importance", color=TEXT, pad=10)
            ax.set_xlabel("Mean |SHAP value|", color=TEXT)
            fig.tight_layout()
            result.figures["shap_bar"] = fig
        except Exception as e:
            result.log.append(f"⚠️ SHAP failed: {e}. Using feature importances.")
            result.shap_available = False

    if not result.top_features and training.feature_importances is not None:
        fi = training.feature_importances
        result.top_features = [(f, round(float(v), 4)) for f, v in fi.head(15).items()]

    # Plain English
    lines = []
    m, pt = training.metrics, training.problem_type
    if pt == "classification":
        acc = m.get("accuracy", 0)
        f1 = m.get("f1_weighted", 0)
        perf = "excellent" if acc >= 0.90 else "good" if acc >= 0.75 else "moderate"
        lines.append(f"The model achieves **{acc*100:.1f}% accuracy** with an F1 score of **{f1:.3f}** — {perf} performance overall.")
    else:
        r2 = m.get("r2", 0)
        rmse = m.get("rmse", 0)
        lines.append(f"The model explains **{max(0,r2)*100:.1f}% of variance** (R²={r2:.3f}), with average prediction error of **{rmse:.3f}** (RMSE).")

    if result.top_features:
        top3 = [f for f, _ in result.top_features[:3]]
        feat_str = ", ".join(f"**{f}**" for f in top3)
        lines.append(f"The most influential features were {feat_str}.")

    if eda.target_column:
        lines.append(f"Trained to predict **'{eda.target_column}'** using {len(eda.feature_columns)} features.")

    if eda.is_imbalanced:
        lines.append(f"⚠️ Class imbalance detected (ratio {eda.imbalance_ratio}:1) — minority class results may be less reliable.")

    if pt == "classification" and m.get("accuracy", 0) > 0.97:
        lines.append("🔎 Accuracy above 97% — verify there's no data leakage.")

    result.plain_english = lines
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineState:
    raw_df: Optional[pd.DataFrame] = None
    clean_df: Optional[pd.DataFrame] = None
    cleaning_report: Optional[CleaningReport] = None
    eda_result: Optional[EDAResult] = None
    selection_result: Optional[ModelSelectionResult] = None
    training_result: Optional[TrainingResult] = None
    explainability_result: Optional[ExplainabilityResult] = None
    current_step: int = 0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""


def run_pipeline(df: pd.DataFrame, target_col: Optional[str] = None, progress_cb: Optional[Callable] = None) -> PipelineState:
    cb = progress_cb or (lambda step, msg: None)
    state = PipelineState(raw_df=df.copy())

    cb(1, "🧹 Step 1 — Data Cleaning…")
    try:
        clean_df, report = run_cleaning(df)
        state.clean_df, state.cleaning_report = clean_df, report
        state.current_step = 1
        if len(clean_df) / max(len(df), 1) < 0.3:
            state.warnings.append("⚠️ Cleaning removed >70% of rows. Results may be unreliable.")
        if len(clean_df) < 20:
            state.aborted, state.abort_reason = True, "Fewer than 20 rows after cleaning — too small to model."
            return state
    except Exception as e:
        state.aborted, state.abort_reason = True, f"Cleaning failed: {e}"
        return state

    cb(2, "🔬 Step 2 — EDA…")
    try:
        eda = run_eda(clean_df, target_col)
        state.eda_result = eda
        state.current_step = 2
        if eda.problem_type == "clustering":
            state.warnings.append("No clear target found. Stopping before model training.")
            return state
        if not eda.target_column:
            state.aborted, state.abort_reason = True, "Could not identify a target column."
            return state
    except Exception as e:
        state.aborted, state.abort_reason = True, f"EDA failed: {e}"
        return state

    cb(3, "⚡ Step 3 — Benchmarking models…")
    try:
        selection = run_model_selection(clean_df, eda)
        state.selection_result = selection
        state.current_step = 3
        best = selection.benchmarks[0].mean_score if selection.benchmarks else 0
        if best < 0.3:
            state.warnings.append(f"⚠️ Best CV score is {best:.3f}. Consider more data or feature engineering.")
    except Exception as e:
        state.aborted, state.abort_reason = True, f"Model selection failed: {e}"
        return state

    cb(4, "🎯 Step 4 — Training & evaluation…")
    try:
        training = run_training(clean_df, eda, selection)
        state.training_result = training
        state.current_step = 4
    except Exception as e:
        state.aborted, state.abort_reason = True, f"Training failed: {e}"
        return state

    cb(5, "🧠 Step 5 — Explainability…")
    try:
        explain = run_explainability(clean_df, eda, training)
        state.explainability_result = explain
        state.current_step = 5
    except Exception as e:
        state.errors.append(f"Explainability failed: {e}")

    cb(5, "✅ Pipeline complete!")
    return state


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AutoML Pipeline", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #060810; color: #C8D6E5; }
.stApp { background-color: #060810; }
h1,h2,h3 { font-family: 'IBM Plex Sans', sans-serif; color: #F0F4F8 !important; }
[data-testid="metric-container"] { background: #0D1117; border: 1px solid #1C2130; border-radius: 10px; padding: 14px 18px; }
[data-testid="metric-container"] label { color: #6B7A8D !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #00E5A0 !important; font-family: 'IBM Plex Mono', monospace !important; }
.stTabs [data-baseweb="tab-list"] { background: #0D1117; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #6B7A8D; border-radius: 6px; border: none; padding: 8px 18px; font-weight: 600; font-size: 13px; }
.stTabs [aria-selected="true"] { background: #1C2130 !important; color: #00E5A0 !important; }
.stButton>button { background: #00E5A0 !important; color: #060810 !important; border: none !important; border-radius: 8px !important; font-weight: 800 !important; font-size: 15px !important; padding: 12px 28px !important; width: 100% !important; }
.stButton>button:hover { background: #00FFB2 !important; }
[data-testid="stFileUploader"] { background: #0D1117; border: 2px dashed #1C2130; border-radius: 10px; padding: 20px; }
.streamlit-expanderHeader { background: #0D1117 !important; color: #C8D6E5 !important; border: 1px solid #1C2130 !important; border-radius: 8px !important; }
.streamlit-expanderContent { background: #0A0C10 !important; border: 1px solid #1C2130 !important; }
.stProgress > div > div { background: #00E5A0 !important; }
.insight-card { background: #0D1117; border: 1px solid #1C2130; border-left: 3px solid #00E5A0; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 6px 0; color: #C8D6E5; font-size: 14px; line-height: 1.5; }
.warning-card { background: #FFB83008; border: 1px solid #FFB83033; border-left: 3px solid #FFB830; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 6px 0; color: #FFB830; font-size: 13px; }
.plain-english-card { background: linear-gradient(135deg, #00E5A008, #4A9EFF08); border: 1px solid #00E5A033; border-radius: 10px; padding: 18px 20px; margin: 8px 0; color: #C8D6E5; font-size: 15px; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0A0C10", "axes.facecolor": "#0D1117",
    "axes.edgecolor": "#1C2130", "axes.labelcolor": "#C8D6E5",
    "xtick.color": "#C8D6E5", "ytick.color": "#C8D6E5",
    "text.color": "#C8D6E5", "grid.color": "#1C2130",
})

# Header
st.markdown("""
<div style="text-align:center;padding:40px 0 24px">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#00E5A0;letter-spacing:0.15em;margin-bottom:12px">● AUTOML PIPELINE · MULTI-AGENT SYSTEM</div>
  <h1 style="font-size:clamp(28px,5vw,52px);font-weight:900;line-height:1.1;margin-bottom:14px;background:linear-gradient(135deg,#F0F4F8,#6B7A8D);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Upload CSV. Get Intelligence.
  </h1>
  <p style="color:#5A6A7A;font-size:15px;max-width:500px;margin:0 auto;line-height:1.6">
    5 autonomous agents — clean, analyze, select, train, explain. Zero configuration.
  </p>
</div>
""", unsafe_allow_html=True)

# Step badges
steps_ui = [("01","Clean","#FF6B6B"),("02","EDA","#00E5A0"),("03","Select","#FFB830"),("04","Train","#4A9EFF"),("05","Explain","#B47AFF")]
cols = st.columns(5)
for i, (num, name, color) in enumerate(steps_ui):
    with cols[i]:
        st.markdown(f"""<div style="text-align:center;padding:12px 8px;background:#0D1117;border:1px solid #1C2130;border-radius:8px">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{color};font-weight:700">STEP {num}</div>
          <div style="color:#F0F4F8;font-weight:700;font-size:13px;margin-top:3px">{name}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload + config
col_up, col_cfg = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader("Drop your CSV file here", type=["csv"], label_visibility="collapsed")
with col_cfg:
    st.markdown('<p style="color:#6B7A8D;font-size:13px;margin-bottom:4px">Target column (optional)</p>', unsafe_allow_html=True)
    target_hint = st.text_input("t", placeholder="e.g. churn, price, label…", label_visibility="collapsed")

df_input = None
if uploaded:
    try:
        encodings = ["utf-8", "latin-1", "windows-1252", "iso-8859-1", "cp1252"]
        df_input = None
        for enc in encodings:
            try:
                uploaded.seek(0)
                df_input = pd.read_csv(uploaded, encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue
        if df_input is None:
            raise ValueError("Could not decode file. Try saving your CSV as UTF-8 in Excel.")
        with st.expander(f"📋 Preview — {df_input.shape[0]:,} rows × {df_input.shape[1]} columns", expanded=False):
            st.dataframe(df_input.head(50), use_container_width=True, height=240)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df_input.shape[0]:,}")
        c2.metric("Columns", f"{df_input.shape[1]}")
        c3.metric("Missing", f"{df_input.isna().sum().sum():,}")
        c4.metric("Duplicates", f"{df_input.duplicated().sum():,}")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

st.markdown("<br>", unsafe_allow_html=True)
run_clicked = st.button("▶  Run AutoML Pipeline", disabled=(df_input is None))

# ── Pipeline Run ───────────────────────────────────────────────────────────────
if run_clicked and df_input is not None:
    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def cb(step, msg):
        progress_bar.progress(step / 5)
        status_text.markdown(f'<p style="color:#00E5A0;font-family:\'IBM Plex Mono\',monospace;font-size:13px">{msg}</p>', unsafe_allow_html=True)

    with st.spinner(""):
        state = run_pipeline(df_input, target_col=target_hint.strip() or None, progress_cb=cb)

    progress_bar.progress(1.0)

    if state.aborted:
        st.error(f"🛑 {state.abort_reason}")
        st.stop()

    for w in state.warnings:
        st.markdown(f'<div class="warning-card">{w}</div>', unsafe_allow_html=True)

    status_text.markdown('<p style="color:#00E5A0;font-weight:700;font-size:15px">✅ Pipeline complete!</p>', unsafe_allow_html=True)
    st.markdown("---")

    tabs = st.tabs(["🧹 Cleaning", "🔬 EDA", "⚡ Model Race", "🎯 Evaluation", "🧠 Explainability"])

    # TAB 1
    with tabs[0]:
        r = state.cleaning_report
        if r:
            st.markdown("#### Data Cleaning Report")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Original Shape", f"{r.original_shape[0]} × {r.original_shape[1]}")
            c2.metric("Final Shape", f"{r.final_shape[0]} × {r.final_shape[1]}")
            c3.metric("Duplicates Removed", r.duplicates_removed)
            c4.metric("Sparse Cols Dropped", len(r.sparse_columns_dropped))
            if r.missing_handled:
                st.markdown("##### 🩹 Missing Value Treatment")
                st.dataframe(pd.DataFrame([{"Column":c,"Missing":i["count"],"Strategy":i["strategy"],"Fill Value":i["fill_value"]} for c,i in r.missing_handled.items()]), use_container_width=True, hide_index=True)
            if r.outliers_handled:
                st.markdown("##### 📐 Outlier Treatment")
                st.dataframe(pd.DataFrame([{"Column":c,"Outliers Capped":i["count"],"Strategy":i["strategy"],"Bounds":str(i["bounds"])} for c,i in r.outliers_handled.items()]), use_container_width=True, hide_index=True)
            if r.type_fixes:
                st.markdown("##### 🔧 Type Fixes")
                st.dataframe(pd.DataFrame([{"Column":c,"Change":v} for c,v in r.type_fixes.items()]), use_container_width=True, hide_index=True)
            with st.expander("📋 Full Log"):
                for line in r.log:
                    st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#6B7A8D">› {line}</div>', unsafe_allow_html=True)

    # TAB 2
    with tabs[1]:
        eda = state.eda_result
        if eda:
            tc = {"classification":"#00E5A0","regression":"#4A9EFF","clustering":"#FFB830"}.get(eda.problem_type,"#6B7A8D")
            st.markdown(f"""<div style="background:{tc}0A;border:1px solid {tc}33;border-radius:10px;padding:20px 24px;margin-bottom:20px">
              <div style="color:{tc};font-size:11px;font-family:'IBM Plex Mono',monospace;font-weight:700;letter-spacing:0.1em">DETECTED PROBLEM TYPE</div>
              <div style="color:{tc};font-size:32px;font-weight:900;font-family:'IBM Plex Mono',monospace">{eda.problem_type.upper()}</div>
              {"<div style='color:#C8D6E5;font-size:14px;margin-top:6px'>Target: <strong>" + (eda.target_column or "") + "</strong></div>" if eda.target_column else ""}
            </div>""", unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Numeric Features", len(eda.numeric_cols))
            c2.metric("Categorical Features", len(eda.categorical_cols))
            c3.metric("High Cardinality", len(eda.high_cardinality_cols))
            if eda.class_counts:
                st.markdown("##### Class Distribution")
                cc_df = pd.DataFrame(list(eda.class_counts.items()), columns=["Class","Count"]).sort_values("Count",ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8, max(3, len(cc_df)*0.4)), facecolor="#0A0C10")
                ax.set_facecolor("#0D1117")
                ax.barh(cc_df["Class"].astype(str)[::-1], cc_df["Count"][::-1], color=["#00E5A0" if i==0 else "#1A4A3A" for i in range(len(cc_df))][::-1], edgecolor="#1C2130")
                ax.set_title("Class Distribution", color="#C8D6E5")
                ax.tick_params(colors="#C8D6E5")
                for s in ax.spines.values(): s.set_edgecolor("#1C2130")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                if eda.is_imbalanced:
                    st.markdown(f'<div class="warning-card">⚠️ Imbalanced dataset — ratio {eda.imbalance_ratio}:1</div>', unsafe_allow_html=True)
            if eda.correlations:
                st.markdown("##### Feature Correlations")
                st.dataframe(pd.DataFrame([{"Feature":r["feature"],"vs":r["target"],"Correlation":r["value"]} for r in eda.correlations]), use_container_width=True, hide_index=True)
            st.markdown("##### Key Insights")
            for ins in eda.insights:
                st.markdown(f'<div class="insight-card">✦ {ins}</div>', unsafe_allow_html=True)
            for w in eda.warnings:
                st.markdown(f'<div class="warning-card">{w}</div>', unsafe_allow_html=True)

    # TAB 3
    with tabs[2]:
        sel = state.selection_result
        if sel and sel.benchmarks:
            winner = sel.benchmarks[0]
            st.markdown(f"""<div style="background:#00E5A00A;border:1px solid #00E5A033;border-radius:10px;padding:18px 22px;margin-bottom:18px;display:flex;align-items:center;gap:16px">
              <span style="font-size:32px">🏆</span>
              <div>
                <div style="color:#6B7A8D;font-size:11px;font-family:'IBM Plex Mono',monospace;letter-spacing:0.1em">WINNER</div>
                <div style="color:#00E5A0;font-size:22px;font-weight:900;font-family:'IBM Plex Mono',monospace">{winner.name}</div>
                <div style="color:#C8D6E5;font-size:14px;margin-top:4px">{sel.metric} = {winner.mean_score:.4f} ± {winner.std_score:.4f}</div>
              </div>
            </div>""", unsafe_allow_html=True)
            names = [b.name for b in sel.benchmarks]
            scores = [b.mean_score for b in sel.benchmarks]
            stds = [b.std_score for b in sel.benchmarks]
            fig, ax = plt.subplots(figsize=(8, max(3, len(names)*0.55)), facecolor="#0A0C10")
            ax.set_facecolor("#0D1117")
            ax.barh(names[::-1], scores[::-1], xerr=stds[::-1],
                    color=["#00E5A0" if i==0 else "#1A3A2A" for i in range(len(names))][::-1],
                    edgecolor="#1C2130", error_kw={"ecolor":"#3D4A5C","capsize":4})
            for i, (score, name) in enumerate(zip(scores[::-1], names[::-1])):
                ax.text(score + 0.005, i, f"{score:.4f}", va="center", color="#C8D6E5", fontsize=10)
            ax.set_title(f"Model Comparison ({sel.metric})", color="#C8D6E5")
            ax.tick_params(colors="#C8D6E5")
            for s in ax.spines.values(): s.set_edgecolor("#1C2130")
            ax.set_xlim(0, max(scores)*1.2)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.dataframe(pd.DataFrame([{"Model":b.name, sel.metric:b.mean_score, "Std":b.std_score, "Rank":f"#{i+1}"} for i,b in enumerate(sel.benchmarks)]), use_container_width=True, hide_index=True)

    # TAB 4
    with tabs[3]:
        tr = state.training_result
        if tr:
            st.markdown(f"#### {tr.model_name} — Evaluation")
            mcols = st.columns(len(tr.metrics))
            for i,(k,v) in enumerate(tr.metrics.items()):
                mcols[i].metric(k.upper(), f"{v:.4f}")
            figs = list(tr.figures.items())
            if len(figs) >= 2:
                c1,c2 = st.columns(2)
                c1.pyplot(figs[0][1]); plt.close(figs[0][1])
                c2.pyplot(figs[1][1]); plt.close(figs[1][1])
                if len(figs) >= 3:
                    st.pyplot(figs[2][1]); plt.close(figs[2][1])
            elif len(figs) == 1:
                st.pyplot(figs[0][1]); plt.close(figs[0][1])
            if tr.classification_report_str:
                with st.expander("📋 Full Classification Report"):
                    st.code(tr.classification_report_str)
            with st.expander("📋 Training Log"):
                for line in tr.log:
                    st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#6B7A8D">› {line}</div>', unsafe_allow_html=True)

    # TAB 5
    with tabs[4]:
        ex = state.explainability_result
        if ex:
            st.markdown("#### 🧠 What the Model Found")
            for sentence in ex.plain_english:
                st.markdown(f'<div class="plain-english-card">{sentence}</div>', unsafe_allow_html=True)
            if "shap_bar" in ex.figures:
                st.markdown("##### SHAP Feature Importance")
                st.pyplot(ex.figures["shap_bar"]); plt.close(ex.figures["shap_bar"])
            elif "feature_importance" in (state.training_result.figures if state.training_result else {}):
                st.pyplot(state.training_result.figures["feature_importance"])
            if ex.top_features:
                st.markdown("##### Feature Rankings")
                df_fi = pd.DataFrame(ex.top_features, columns=["Feature","Importance Score"])
                df_fi.index += 1
                st.dataframe(df_fi, use_container_width=True)
            method = "SHAP TreeExplainer" if ex.shap_available else "Model Feature Importances"
            st.markdown(f'<p style="color:#3D4A5C;font-size:12px;font-family:\'IBM Plex Mono\',monospace;margin-top:16px">Method: {method}</p>', unsafe_allow_html=True)

    # Downloads
    st.markdown("---")
    st.markdown("### 💾 Download Results")
    dl1, dl2 = st.columns(2)
    with dl1:
        if state.clean_df is not None:
            buf = io.StringIO()
            state.clean_df.to_csv(buf, index=False)
            st.download_button("⬇ Download Cleaned CSV", data=buf.getvalue(), file_name="cleaned_data.csv", mime="text/csv")
    with dl2:
        if state.explainability_result and state.training_result:
            lines = ["# AutoML Pipeline Report\n", f"## Model: {state.training_result.model_name}\n",
                     f"## Problem Type: {state.eda_result.problem_type if state.eda_result else 'N/A'}\n",
                     "\n## Metrics\n"]
            for k,v in (state.training_result.metrics or {}).items():
                lines.append(f"- **{k.upper()}**: {v}\n")
            lines.append("\n## Key Insights\n")
            for line in state.explainability_result.plain_english:
                lines.append(f"- {line.replace('**','')}\n")
            st.download_button("⬇ Download Report (MD)", data="".join(lines), file_name="automl_report.md", mime="text/markdown")

if not uploaded:
    st.markdown("""<div style="text-align:center;padding:48px 0;opacity:0.4">
      <div style="font-size:52px;margin-bottom:12px">🧬</div>
      <div style="color:#6B7A8D;font-size:14px;font-weight:600">Upload a CSV to begin</div>
      <div style="color:#3D4A5C;font-size:12px;margin-top:6px">Works with classification, regression, or clustering datasets</div>
    </div>""", unsafe_allow_html=True)