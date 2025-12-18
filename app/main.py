from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config.configuracao import settings
from app.security.seguranca import configurar_middlewares
from app.api.v1.router import api_router
from app.services.model_loader import ModelLoader


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"--- [STARTUP] Inicializando {settings.PROJECT_TITLE} ---")
    try:
        loader = ModelLoader.get_instance()
        print(f"--- [STARTUP] Modelos carregados: {list(loader.models.keys())} ---")
    except Exception as e:
        print(f"--- [ERRO CRÍTICO] Falha ao carregar modelos: {e} ---")

    yield

    print("--- [SHUTDOWN] Desligando API ---")


app = FastAPI(
    title=settings.PROJECT_TITLE,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)

configurar_middlewares(app)

app.include_router(api_router, prefix=settings.API_V1_STR)