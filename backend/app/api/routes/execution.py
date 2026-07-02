from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.agents.execution_setup import build_setup_steps_text, inject_setup_steps
from app.agents.step_normalizer import normalize_feature_text
from app.core.config import get_settings
from app.core.database import SessionLocal, get_db
from app.models.execution import Execution, ExecutionFeatureFile, ExecutionStep
from app.schemas.execution import (
    ExecutionFeatureFileResponse,
    ExecutionResponse,
    ExecutionRunRequest,
    ExecutionStepResponse,
)
from app.services import report_service, video_service
from app.services.execution_service import execute_execution_run

router = APIRouter(prefix="/execution", tags=["execution"])


def _run_in_background(execution_id: str, payload: ExecutionRunRequest) -> None:
    db = SessionLocal()
    try:
        execute_execution_run(
            db,
            execution_id,
            base_url=payload.base_url,
            feature_files=[f.model_dump() for f in payload.feature_files],
            failure_mode=payload.failure_mode,
            headless=payload.headless,
        )
    finally:
        db.close()


@router.post("/run", response_model=ExecutionResponse)
def start_execution(payload: ExecutionRunRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if not payload.feature_files:
        raise HTTPException(status_code=400, detail="At least one feature file is required")

    # Every feature file gets its own fresh Edge session (see
    # execution_orchestrator.run_execution), so the login/search setup steps are
    # injected into each one individually, not just the first.
    setup_text = build_setup_steps_text(
        payload.username, payload.password, payload.transaction_number, payload.transaction_name
    )
    if setup_text:
        for feature_file in payload.feature_files:
            feature_file.raw_text = inject_setup_steps(feature_file.raw_text, setup_text)

    # Universal step normalization: any phrasing is compiled into the canonical
    # Step DSL (alias patterns first, one batched LLM call for leftovers) so an
    # arbitrary uploaded feature file executes. Unmapped lines are deliberately
    # left untouched — they surface as loud `undefined_step` failures in the
    # report rather than being silently skipped or guessed at.
    for feature_file in payload.feature_files:
        feature_file.raw_text = normalize_feature_text(feature_file.raw_text).canonical_text

    execution = Execution(base_url=payload.base_url, failure_mode=payload.failure_mode, status="pending")
    db.add(execution)
    db.commit()
    db.refresh(execution)

    for sequence, feature_file in enumerate(payload.feature_files, start=1):
        db.add(
            ExecutionFeatureFile(
                execution_id=execution.id,
                sequence=sequence,
                filename=feature_file.filename,
                raw_text=feature_file.raw_text,
                status="pending",
            )
        )
    db.commit()

    background_tasks.add_task(_run_in_background, execution.id, payload)
    return execution


@router.post("/normalize-preview")
def normalize_preview(payload: ExecutionRunRequest):
    """Dry-run the universal step normalizer: shows exactly how each uploaded
    feature file's lines will be interpreted (original -> canonical -> method)
    without launching a browser. Lets a tester review the translation before
    running against a live banking system."""
    if not payload.feature_files:
        raise HTTPException(status_code=400, detail="At least one feature file is required")
    previews = []
    for feature_file in payload.feature_files:
        result = normalize_feature_text(feature_file.raw_text)
        previews.append(
            {
                "filename": feature_file.filename,
                "canonical_text": result.canonical_text,
                "fully_mapped": result.fully_mapped,
                "unmapped": result.unmapped,
                "mappings": [
                    {"original": m.original, "canonical": m.canonical, "method": m.method}
                    for m in result.mappings
                ],
            }
        )
    return {"previews": previews}


@router.get("/{execution_id}", response_model=ExecutionResponse)
def get_execution(execution_id: str, db: Session = Depends(get_db)):
    execution = db.get(Execution, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution


@router.get("/{execution_id}/feature-files", response_model=list[ExecutionFeatureFileResponse])
def get_execution_feature_files(execution_id: str, db: Session = Depends(get_db)):
    return (
        db.query(ExecutionFeatureFile)
        .filter(ExecutionFeatureFile.execution_id == execution_id)
        .order_by(ExecutionFeatureFile.sequence)
        .all()
    )


@router.get("/{execution_id}/steps", response_model=list[ExecutionStepResponse])
def get_execution_steps(execution_id: str, db: Session = Depends(get_db)):
    return (
        db.query(ExecutionStep)
        .filter(ExecutionStep.execution_id == execution_id)
        .order_by(ExecutionStep.sequence)
        .all()
    )


@router.get("/{execution_id}/report")
def get_execution_report(execution_id: str, format: str = "html", db: Session = Depends(get_db)):
    if db.get(Execution, execution_id) is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    if format == "html":
        path = report_service.render_html_report(db, execution_id)
        return FileResponse(path, media_type="text/html", filename=path.name)
    if format == "json":
        path = report_service.render_json_report(db, execution_id)
        return FileResponse(path, media_type="application/json", filename=path.name)
    if format == "pdf":
        path = report_service.render_pdf_report(db, execution_id)
        return FileResponse(path, media_type="application/pdf", filename=path.name)
    raise HTTPException(status_code=400, detail="format must be 'html', 'json', or 'pdf'")


@router.get("/{execution_id}/video")
def get_execution_video(execution_id: str, db: Session = Depends(get_db)):
    if db.get(Execution, execution_id) is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    try:
        path = video_service.render_execution_video(db, execution_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/screenshot")
def get_screenshot(path: str):
    settings = get_settings()
    export_root = Path(settings.export_dir).resolve()
    requested = Path(path).resolve()
    if export_root not in requested.parents and requested != export_root:
        raise HTTPException(status_code=400, detail="Invalid screenshot path")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(requested, media_type="image/png")
