from backend.app.ml.inference.predictor import Predictor
from backend.app.llm.client import LLMAnalyst

# Instantiated singletons
predictor = Predictor()
llm_analyst = LLMAnalyst()

def get_predictor() -> Predictor:
    return predictor

def get_llm_analyst() -> LLMAnalyst:
    return llm_analyst
