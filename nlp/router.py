from __future__ import annotations
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nlp.nlp import Trainer

app = FastAPI()
trainer = Trainer()
#BaseModel is used as data validator when using fast api it cares all about exception handilng and validate
#your incoming json to be what you want to be.

class TestingData(BaseModel):
    texts: List[str]

class QueryText(BaseModel):
    text: str

class StatusObject(BaseModel):
    status: str
    timestamp: str
    classes: List[str]
    evaluation: Dict

class PredictionObject(BaseModel):
    text: str
    predictions: Dict

class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]


@app.get("/status", summary="Get current status of the system")
def get_status():
    status = trainer.get_status()
    return StatusObject(**status)

@app.get("/trainMachineLearning", summary="Train a new Machine Learning model")
def train():
    try:
        trainer.trainMachineLearning(trainer.merge().text, trainer.merge().dialect)
        status = trainer.get_status()
        return StatusObject(**status)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/trainDeepLearning", summary="Train a new Deep Learning model")
def train():
    try:
        trainer.trainDeepLearning(trainer.merge().text, trainer.merge().dialect)
        status = trainer.get_status()
        return StatusObject(**status)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict", summary="Predict single input")
def predict(query_text: QueryText):
    try:
        prediction = trainer.predict([query_text.text])[0]
        return PredictionObject(**prediction)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict-batch", summary="predict a batch of sentences")
def predict_batch(testing_data:TestingData):
    try:
        predictions = trainer.predict(testing_data.texts)
        return PredictionsObject(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/")
def home():
    return({"message": "System is up"})
