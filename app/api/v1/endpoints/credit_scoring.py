from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.services.predictors.financial_predictor import FinancialRiskPredictor

router = APIRouter()

class CreditRequest(BaseModel):
    income: float
    debt_ratio: float
    loan_amount: float
    late_payments_count: int
    user_id: str

class CreditResponse(BaseModel):
    status: str
    risk_score: float
    action_taken: str
    message: str

def _approve_loan_in_core_banking(user_id, amount):
    print(f"$$$ [CORE BANKING] Empréstimo de {amount} APROVADO para {user_id}. Dinheiro liberado.")


def _send_to_manual_review_queue(user_id, risk_score):
    print(f"!!! [HUMAN REVIEW] Usuário {user_id} enviado para análise de mesa. Risco: {risk_score:.2f}")


def _auto_reject_loan(user_id, reason):
    print(f"XXX [REJECTION] Empréstimo negado para {user_id}. Motivo: {reason}")


@router.post("/analyze", response_model=CreditResponse)
async def analyze_credit_request(request: CreditRequest, background_tasks: BackgroundTasks):

    predictor = FinancialRiskPredictor.get_instance()

    risk_prob = predictor.predict_risk_probability(request.dict())

    SAFE_THRESHOLD = 0.3
    CRITICAL_THRESHOLD = 0.8

    result = CreditResponse(
        status="PROCESSING",
        risk_score=risk_prob,
        action_taken="NONE",
        message=""
    )

    if risk_prob < SAFE_THRESHOLD:

        result.status = "APPROVED"
        result.action_taken = "AUTO_EXECUTION"
        result.message = "Crédito aprovado automaticamente pela IA."

        _approve_loan_in_core_banking(request.user_id, request.loan_amount)

    elif risk_prob >= CRITICAL_THRESHOLD:
        result.status = "REJECTED"
        result.action_taken = "AUTO_BLOCK"
        result.message = "Solicitação negada por alto risco estatístico."

        _auto_reject_loan(request.user_id, "High Probability of Default detected by ML Model")

    else:
        result.status = "PENDING_REVIEW"
        result.action_taken = "QUEUED_FOR_HUMAN"
        result.message = "Solicitação enviada para análise de um especialista."

        background_tasks.add_task(_send_to_manual_review_queue, request.user_id, risk_prob)

    return result