from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import PipelineConfig
from ..service import RagService
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PipelineConfig.from_yaml("configs/pipeline.yaml")
    app.state.service = RagService.build(config)
    yield


app = FastAPI(title="Edgar RAG Pipeline API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
