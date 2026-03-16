"""
Orchestrator — routes data between agents, handles loop-back decisions.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable

from agents.cleaner import DataCleaningAgent, CleaningReport
from agents.eda import EDAAgent, EDAResult
from agents.model_selector import ModelSelectionAgent, ModelSelectionResult
from agents.trainer import TrainingAgent, TrainingResult
from agents.insights import ExplainabilityAgent, ExplainabilityResult  


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


class Orchestrator:
    """
    Runs each agent in sequence and handles:
    - Loop-back: if data is too sparse after cleaning
    - Abort: if no viable target or too few rows
    - Progress callbacks for Streamlit UI updates
    """

    def __init__(self, target_col: Optional[str] = None, progress_cb: Optional[Callable] = None):
        self.target_col = target_col
        self.progress_cb = progress_cb or (lambda step, msg: None)

    def run(self, df: pd.DataFrame) -> PipelineState:
        state = PipelineState(raw_df=df.copy())

        # ── Step 1 · Clean ─────────────────────────────────────────────────────
        self.progress_cb(1, "🧹 Running Data Cleaning Agent…")
        try:
            agent = DataCleaningAgent()
            clean_df, report = agent.run(df)
            state.clean_df = clean_df
            state.cleaning_report = report
            state.current_step = 1

            # Loop-back check: too many rows dropped
            row_retention = len(clean_df) / max(len(df), 1)
            if row_retention < 0.3:
                state.warnings.append(
                    f"⚠️ Cleaning removed {(1-row_retention)*100:.0f}% of rows. Results may be unreliable."
                )

            if len(clean_df) < 20:
                return self._abort(state, "Dataset has fewer than 20 rows after cleaning — too small to model.")

        except Exception as e:
            return self._abort(state, f"Cleaning agent failed: {e}")

        # ── Step 2 · EDA ───────────────────────────────────────────────────────
        self.progress_cb(2, "🔬 Running EDA Agent…")
        try:
            agent = EDAAgent(target_col=self.target_col)
            eda = agent.run(clean_df)
            state.eda_result = eda
            state.current_step = 2

            if eda.problem_type == "clustering":
                state.warnings.append(
                    "No clear target column found — pipeline will stop before model selection."
                )
                return state  # clustering support can be added later

            if not eda.target_column:
                return self._abort(state, "Could not identify a target column.")

        except Exception as e:
            return self._abort(state, f"EDA agent failed: {e}")

        # ── Step 3 · Model Selection ───────────────────────────────────────────
        self.progress_cb(3, "⚡ Benchmarking models…")
        try:
            agent = ModelSelectionAgent()
            selection = agent.run(clean_df, eda)
            state.selection_result = selection
            state.current_step = 3

            # Check if best score is too low
            best_score = selection.benchmarks[0].mean_score if selection.benchmarks else 0
            if best_score < 0.3:
                state.warnings.append(
                    f"⚠️ Best CV score is only {best_score:.3f}. "
                    "Consider feature engineering or getting more data."
                )

        except Exception as e:
            return self._abort(state, f"Model selection failed: {e}")

        # ── Step 4 · Training ──────────────────────────────────────────────────
        self.progress_cb(4, "🎯 Training & evaluating best model…")
        try:
            agent = TrainingAgent(tune=True)
            training = agent.run(clean_df, eda, selection)
            state.training_result = training
            state.current_step = 4
        except Exception as e:
            return self._abort(state, f"Training agent failed: {e}")

        # ── Step 5 · Explainability ────────────────────────────────────────────
        self.progress_cb(5, "🧠 Computing SHAP values & generating insights…")
        try:
            agent = ExplainabilityAgent()
            explain = agent.run(clean_df, eda, training)
            state.explainability_result = explain
            state.current_step = 5
        except Exception as e:
            state.errors.append(f"Explainability agent failed: {e}")
            # Non-fatal — still return results

        self.progress_cb(5, "✅ Pipeline complete!")
        return state

    def _abort(self, state: PipelineState, reason: str) -> PipelineState:
        state.aborted = True
        state.abort_reason = reason
        state.errors.append(reason)
        return state
