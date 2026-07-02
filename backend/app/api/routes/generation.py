"""Feature Generation routes.

POST /generation/draft            — no browser is launched: loads the transaction's
                                    effective Locator Repository, drafts a feature
                                    file (LLM if configured, deterministic builder
                                    otherwise), validates it, and repairs field-name
                                    mismatches. Returns the draft for HUMAN REVIEW —
                                    the safe default for a banking system.

POST /generation/generate-and-run — the full loop: draft → validate → repair →
                                    execute against the given base_url → diagnose
                                    failures → repair → retry (bounded). Runs
                                    synchronously and returns the final feature text,
                                    per-step results, and the graph's audit notes.
                                    Application-level failures (assertion/app_error)
                                    are never blindly retried.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.generation import GenerateAndRunRequest, GenerationRequest, GenerationResponse

router = APIRouter(prefix="/generation", tags=["generation"])


def _build_graph(db: Session):
    try:
        from app.agents.feature_generation_graph import FeatureGenerationGraph
        return FeatureGenerationGraph(db)
    except RuntimeError as exc:  # langgraph missing — surface a clear message
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/draft", response_model=GenerationResponse)
def draft_feature(payload: GenerationRequest, db: Session = Depends(get_db)):
    graph = _build_graph(db)
    result = graph.run(
        transaction_number=payload.transaction_number,
        intent=payload.intent,
        field_values=payload.field_values,
        needs_approval=payload.needs_approval,
        maker_username=payload.maker_username,
        maker_password=payload.maker_password,
        checker_username=payload.checker_username,
        checker_password=payload.checker_password,
        execute=False,
    )
    return GenerationResponse(**result)


@router.post("/generate-and-run", response_model=GenerationResponse)
def generate_and_run(payload: GenerateAndRunRequest, db: Session = Depends(get_db)):
    graph = _build_graph(db)
    result = graph.run(
        transaction_number=payload.transaction_number,
        intent=payload.intent,
        field_values=payload.field_values,
        needs_approval=payload.needs_approval,
        maker_username=payload.maker_username,
        maker_password=payload.maker_password,
        checker_username=payload.checker_username,
        checker_password=payload.checker_password,
        execute=True,
        base_url=payload.base_url,
        headless=payload.headless,
    )
    return GenerationResponse(**result)
