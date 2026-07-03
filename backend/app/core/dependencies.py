from backend.app.ml.inference.predictor import Predictor
from backend.app.llm.client import LLMAnalyst

predictor = None
llm_analyst = None

def get_predictor() -> Predictor:
    global predictor
    if predictor is None:
        predictor = Predictor()
    return predictor

def get_llm_analyst() -> LLMAnalyst:
    global llm_analyst
    if llm_analyst is None:
        llm_analyst = LLMAnalyst()
    return llm_analyst
