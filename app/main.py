from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import health_diagnosis
from app.services.model_loader import ModelLoader

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- [STARTUP] Inicializando Ortzion AI API ---")

    try:
        loader = ModelLoader.get_instance()
        print(f"--- [STARTUP] Modelos carregados: {list(loader.models.keys())} ---")
    except Exception as e:
        print(f"--- [ERRO CRÍTICO] Falha ao carregar modelos: {e} ---")

    yield

    print("--- [SHUTDOWN] Desligando API ---")


app = FastAPI(
    title="Ortzion AI - Enterprise MLOps API",
    description="API para inferência de modelos de Inteligência Artificial (Health & Finance)",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    health_diagnosis.router,
    prefix="/api/v1/health",
    tags=["Health Diagnosis"]
)


@app.get("/healthcheck", tags=["System"])
def healthcheck():
    loader = ModelLoader.get_instance()
    return {
        "status": "online",
        "system": "Ortzion AI",
        "models_active": list(loader.models.keys())
    }