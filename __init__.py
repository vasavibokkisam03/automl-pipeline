# agents/__init__.py - Auto-imports all agents for easy access
 
from agents.cleaner import DataCleaningAgent, CleaningReport
from agents.eda import EDAAgent, EDAResult
from agents.model_selector import ModelSelectionAgent, ModelSelectionResult, ModelBenchmark
from agents.trainer import TrainingAgent, TrainingResult
from agents.insights import ExplainabilityAgent, ExplainabilityResult
 
__all__ = [
    "DataCleaningAgent", "CleaningReport",
    "EDAAgent", "EDAResult",
    "ModelSelectionAgent", "ModelSelectionResult", "ModelBenchmark",
    "TrainingAgent", "TrainingResult",
    "ExplainabilityAgent", "ExplainabilityResult",
]
 
 