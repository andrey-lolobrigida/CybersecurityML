"""
API route definitions.
"""

from fastapi import APIRouter, Request

from api.models import PredictionResponse, SessionInput
from src.main import run

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(session: SessionInput, request: Request):
    row = session.model_dump()
    return run(row, request.app.state.pipeline)