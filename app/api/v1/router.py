from fastapi import APIRouter
from app.api.v1.endpoints import health_diagnosis, credit_scoring

api_router = APIRouter()

api_router.include_router(
    health_diagnosis.router,
    prefix="/health",
    tags=["Health Diagnosis"]
)

api_router.include_router(
    credit_scoring.router,
    prefix="/credit",
    tags=["Credit Scoring"]
)