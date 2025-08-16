from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib, json, os

app = FastAPI(title="ML Inference API", version="1.0")

MODEL_PATH = os.path.join("artifacts", "model.joblib")
METRICS_PATH = os.path.join("artifacts", "metrics.json")
pipe = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    META = json.load(f)

class Payload(BaseModel):
    features: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "model": META.get("model"), "task": META.get("task"), "target": META.get("target")}

@app.post("/predict")
def predict(payload: Payload):
    try:
        X = pd.DataFrame([payload.features])
        y_pred = pipe.predict(X)
        # If classification and proba available
        proba = None
        try:
            proba = pipe.predict_proba(X)
            proba = proba.tolist()
        except Exception:
            pass
        return {"prediction": y_pred.tolist(), "proba": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

