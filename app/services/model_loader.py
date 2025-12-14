# app/services/model_loader.py
import torch
import joblib
import os
from torch import nn


# --- Definição da Arquitetura Neural (Deve ser idêntica ao treino) ---
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


class ModelLoader:
    _instance = None

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self._load_models()

    def _load_models(self):
        print("--- [SISTEMA] Carregando Modelos na Memória ---")
        base_path = "models_store/health"

        # 1. Carregar Scaler (Fundamental)
        scaler_path = os.path.join(base_path, "scaler_v2.pkl")
        if os.path.exists(scaler_path):
            self.scalers["iris_scaler"] = joblib.load(scaler_path)
            print(f"[OK] Scaler carregado: {scaler_path}")

        # 2. Carregar Rede Neural (PyTorch)
        model_path = os.path.join(base_path, "model_v2.pth")  # Ajuste o nome conforme sua pasta
        if os.path.exists(model_path):
            net = IrisNet()
            net.load_state_dict(torch.load(model_path, map_location='cpu'))
            net.eval()  # Modo de inferência
            self.models["iris_neural"] = net
            print(f"[OK] Rede Neural carregada: {model_path}")
        else:
            print(f"[ERRO] Modelo não encontrado: {model_path}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Variável global para acesso fácil
loader = ModelLoader.get_instance()