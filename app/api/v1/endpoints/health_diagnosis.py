# app/api/v1/endpoints/health_diagnosis.py
from fastapi import APIRouter, HTTPException
from app.schemas.health import IrisInput, IrisPrediction
from app.services.predictors.torch_predictor import HealthPredictor
from app.services.model_loader import ModelLoader

router = APIRouter()


@router.post("/diagnose/iris", response_model=IrisPrediction)
def predict_iris_species(data: IrisInput):
    """
    Realiza o diagnóstico da espécie da flor Iris usando Deep Learning.
    """
    try:
        # Extrai os dados do Pydantic para uma lista simples
        features = [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]

        # Chama a camada de serviço (Strategy Pattern implícito)
        result = HealthPredictor.predict_iris(features)

        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/healthcheck")
def healthcheck():
    loader = ModelLoader.get_instance()
    return {
        "status": "online",
        "system": "Ortzion AI",
        "models_active": list(loader.models.keys())
    }