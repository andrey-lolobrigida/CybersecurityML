"""
FastAPI application factory.

Run with:
    uv run uvicorn api.app:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from src.main import build_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = build_pipeline()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cybersecurity Intrusion Detection API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()