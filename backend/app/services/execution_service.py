import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.agents.execution_orchestrator import run_execution
from app.agents.failure_analysis_agent import classify_failure
from app.agents.knowledge_agent import (
    record_execution_episode,
    record_failure_lesson,
    record_healing_lesson,
)
from app.agents.self_healing_agent import explain_healing
from app.core.config import get_settings
from app.models.execution import Execution, ExecutionFeatureFile, ExecutionStep, Failure, HealingHistory
from app.models.locator import LocatorEntry

logger = logging.getLogger(__name__)


def _apply_healing_to_repository(db: Session, healing_info: dict, field_name: str, transaction_number: str | None) -> None:
    entry_id = healing_info.get("entry_id")
    if entry_id:
        entry = db.get(LocatorEntry, entry_id)
        if entry is not None:
            entry.fallback_locator = entry.priority_locator
            entry.fallback_locator_type = entry.priority_locator_type
            entry.priority_locator = healing_info["new_locator"]
            entry.priority_locator_type = healing_info["new_locator_type"]
            entry.ai_confidence_score = healing_info.get("confidence", entry.ai_confidence_score)
            return

    db.add(
        LocatorEntry(
            transaction_number=transaction_number,
            screen_name="healed",
            field_name=field_name,
            priority_locator=healing_info["new_locator"],
            priority_locator_type=healing_info["new_locator_type"],
            ai_confidence_score=healing_info.get("confidence", 0.5),
            control_type=healing_info.get("control_type"),
        )
    )


def execute_execution_run(
    db: Session,
    execution_id: str,
    base_url: str,
    feature_files: list[dict],
    failure_mode: str,
    headless: bool | None,
) -> None:
    execution = db.get(Execution, execution_id)
    if execution is None:
        return

    execution.status = "running"
    db.commit()

    settings = get_settings()
    screenshot_dir = Path(settings.export_dir) / "screenshots" / execution_id

    def on_feature_file_status(filename: str, status: str) -> None:
        row = (
            db.query(ExecutionFeatureFile)
            .filter(ExecutionFeatureFile.execution_id == execution_id, ExecutionFeatureFile.filename == filename)
            .first()
        )
        if row:
            row.status = status
            db.commit()

    def on_step_complete(result: dict) -> None:
        step = result["step"]
        step_row = ExecutionStep(
            execution_id=execution_id,
            feature_filename=result["feature_filename"],
            sequence=result["sequence"],
            step_text=step.raw,
            step_type=type(step).__name__,
            status=result["status"],
            locator_used=result["locator_used"],
            healed=result["healed"],
            error_message=str(result["error"]) if result["error"] else None,
            screenshot_before=result["screenshot_before"],
            screenshot_after=result["screenshot_after"],
            duration_ms=result["duration_ms"],
        )
        db.add(step_row)
        db.flush()

        if result["healed"] and result["healing_info"]:
            healing_info = result["healing_info"]
            explanation = explain_healing(
                getattr(step, "field_name", getattr(step, "value", "field")),
                healing_info.get("old_locator"),
                healing_info["new_locator"],
            )
            db.add(
                HealingHistory(
                    execution_step_id=step_row.id,
                    field_name=getattr(step, "field_name", getattr(step, "value", "field")),
                    old_locator=healing_info.get("old_locator"),
                    new_locator=healing_info["new_locator"],
                    new_locator_type=healing_info["new_locator_type"],
                    confidence=healing_info.get("confidence", 0.0),
                    success=True,
                    llm_explanation=explanation,
                )
            )
            _apply_healing_to_repository(
                db, healing_info, getattr(step, "field_name", getattr(step, "value", "field")), None
            )
            try:
                record_healing_lesson(
                    db,
                    transaction_number=None,
                    field_name=getattr(step, "field_name", getattr(step, "value", "field")),
                    old_locator=healing_info.get("old_locator"),
                    new_locator=healing_info["new_locator"],
                    explanation=explanation,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Knowledge capture (healing lesson) failed; continuing")

        if result["status"] == "failed" and result["error"] is not None:
            analysis = classify_failure(step, result["error"])
            db.add(
                Failure(
                    execution_step_id=step_row.id,
                    category=analysis.category,
                    root_cause=analysis.root_cause,
                    suggested_fix=analysis.suggested_fix,
                    confidence=analysis.confidence,
                )
            )
            try:
                record_failure_lesson(
                    db,
                    transaction_number=None,
                    step_text=step.raw,
                    category=analysis.category,
                    root_cause=analysis.root_cause,
                    suggested_fix=analysis.suggested_fix,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Knowledge capture (failure lesson) failed; continuing")

        execution.total_steps += 1
        if result["status"] == "passed":
            execution.passed_steps += 1
        else:
            execution.failed_steps += 1
        if result["healed"]:
            execution.healed_steps += 1
        db.commit()

    try:
        run_execution(
            base_url=base_url,
            feature_files=feature_files,
            failure_mode=failure_mode,
            db=db,
            screenshot_dir=screenshot_dir,
            on_feature_file_status=on_feature_file_status,
            on_step_complete=on_step_complete,
            headless=headless,
        )
        execution.status = "failed" if execution.failed_steps > 0 else "success"
    except Exception as exc:
        logger.exception("Execution run %s failed", execution_id)
        execution.status = "failed"
        execution.error_message = str(exc)
    finally:
        execution.completed_at = datetime.now(timezone.utc)
        db.commit()
        # Episode memory: one plain-language record of what this run did,
        # retrievable by future generation/repair prompts. Never fatal.
        try:
            step_rows = [
                {
                    "step_text": row.step_text,
                    "status": row.status,
                    "healed": row.healed,
                    "error_message": row.error_message,
                }
                for row in db.query(ExecutionStep)
                .filter(ExecutionStep.execution_id == execution_id)
                .order_by(ExecutionStep.sequence)
                .all()
            ]
            record_execution_episode(db, None, step_rows, execution.status)
            db.commit()
        except Exception:  # noqa: BLE001
            logger.warning("Knowledge capture (episode) failed; continuing")
