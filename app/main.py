import os
import joblib
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Importação da arquitetura da rede neural (se estivesse em outro arquivo)
# from app.architecture import IrisNet

# --- Definição da Arquitetura (Mantida aqui para facilitar sua execução imediata) ---
from torch import nn


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.layer1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# --- Schemas (Contratos de Dados) ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# --- Gestão de Estado Global (State Management) ---
ml_artifacts = {
    "v1_sklearn": None,
    "v2_torch_model": None,
    "v2_scaler": None
}

MODEL_DIR = "models_store"  # Ajuste conforme o caminho real


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    Carrega os modelos na memória RAM apenas uma vez durante o startup.
    """
    print("--- INICIANDO CARREGAMENTO DE MODELOS ---")

    # 1. Carregar Modelo V1 (Scikit-Learn)
    path_v1 = os.path.join(MODEL_DIR, "model_v1.pkl")
    if os.path.exists(path_v1):
        ml_artifacts["v1_sklearn"] = joblib.load(path_v1)
        print(f"[OK] Modelo V1 carregado: {path_v1}")
    else:
        print(f"[AVISO] Modelo V1 não encontrado em: {path_v1}")

    # 2. Carregar Artefatos V2 (PyTorch + Scaler)
    path_scaler = os.path.join(MODEL_DIR, "scaler_v2.pkl")
    path_torch = os.path.join(MODEL_DIR, "model_v2.pth")

    # Carrega Scaler
    if os.path.exists(path_scaler):
        ml_artifacts["v2_scaler"] = joblib.load(path_scaler)
        print(f"[OK] Scaler V2 carregado.")
    else:
        print(f"[ERRO CRÍTICO] Scaler V2 ausente.")

    # Carrega Rede Neural
    if os.path.exists(path_torch):
        try:
            model = IrisNet()
            # map_location='cpu' garante que funcione mesmo se treinado em GPU
            model.load_state_dict(torch.load(path_torch, map_location=torch.device('cpu')))
            model.eval()  # Trava Dropouts e BatchNormalization para inferência
            ml_artifacts["v2_torch_model"] = model
            print(f"[OK] Rede Neural V2 carregada.")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar pesos do PyTorch: {e}")
    else:
        print(f"[AVISO] Pesos V2 não encontrados em: {path_torch}")

    yield  # A aplicação roda aqui

    # Limpeza (Shutdown)
    print("--- LIMPANDO MEMÓRIA ---")
    ml_artifacts.clear()


# Inicialização da App
app = FastAPI(
    title="Iris Species Classifier API",
    description="API para inferência de modelos ML (Sklearn) e DL (PyTorch)",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    """Endpoint para verificar se a API está de pé e quais modelos estão carregados."""
    return {
        "status": "online",
        "models_loaded": {k: v is not None for k, v in ml_artifacts.items()}
    }


@app.post("/predict/v1")
def predict_v1_sklearn(data: IrisInput):
    """Inferência usando Scikit-Learn (Random Forest/Logistic Regression)."""
    model = ml_artifacts["v1_sklearn"]

    if not model:
        raise HTTPException(status_code=503, detail="Modelo V1 não disponível.")

    try:
        features = np.array([[
            data.sepal_length, data.sepal_width,
            data.petal_length, data.petal_width
        ]])

        prediction = model.predict(features)

        # Tenta pegar probabilidade se o modelo suportar
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)
            confidence = float(np.max(probs))

        return {
            "model_version": "v1 (sklearn)",
            "predicted_class": int(prediction[0]),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência V1: {str(e)}")


@app.post("/predict/v2")
def predict_v2_pytorch(data: IrisInput):
    """Inferência usando Rede Neural PyTorch."""
    model = ml_artifacts["v2_torch_model"]
    scaler = ml_artifacts["v2_scaler"]

    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Artefatos V2 (Modelo ou Scaler) indisponíveis.")

    try:
        # 1. Prepara dados e Normaliza
        raw_data = np.array([[
            data.sepal_length, data.sepal_width,
            data.petal_length, data.petal_width
        ]])
        processed_data = scaler.transform(raw_data)

        # 2. Converte para Tensor
        tensor_data = torch.tensor(processed_data, dtype=torch.float32)

        # 3. Inferência (Sem cálculo de gradiente)
        with torch.no_grad():
            logits = model(tensor_data)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        return {
            "model_version": "v2 (pytorch_nn)",
            "predicted_class": int(predicted_class.item()),
            "confidence": float(confidence.item())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência V2: {str(e)}")