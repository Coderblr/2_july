import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.agents.crawl_orchestrator import run_crawl
from app.agents.knowledge_agent import record_screen_knowledge
from app.agents.locator_builder_agent import build_locator_entries
from app.models.locator import CrawlRun, FeatureFileDoc, LocatorEntry, Screen

logger = logging.getLogger(__name__)


def execute_crawl_run(
    db: Session,
    crawl_run_id: str,
    base_url: str,
    username: str | None,
    password: str | None,
    transaction_number: str | None,
    transaction_name: str | None,
    feature_file_ids: list[str] | None,
    headless: bool | None,
) -> None:
    crawl_run = db.get(CrawlRun, crawl_run_id)
    if crawl_run is None:
        return

    crawl_run.status = "running"
    db.commit()

    feature_file_text = None
    if feature_file_ids:
        docs = db.query(FeatureFileDoc).filter(FeatureFileDoc.id.in_(feature_file_ids)).all()
        feature_file_text = "\n\n".join(doc.raw_text for doc in docs) or None

    try:
        result = run_crawl(
            base_url=base_url,
            username=username,
            password=password,
            transaction_number=transaction_number,
            transaction_name=transaction_name,
            feature_file_text=feature_file_text,
            headless=headless,
        )
        _persist_result(db, crawl_run, result, transaction_number)
        crawl_run.status = "success"
    except Exception as exc:
        logger.exception("Crawl run %s failed", crawl_run_id)
        crawl_run.status = "failed"
        crawl_run.error_message = str(exc)
    finally:
        crawl_run.completed_at = datetime.now(timezone.utc)
        db.commit()


def _persist_result(db: Session, crawl_run: CrawlRun, result: dict, transaction_number: str | None) -> None:
    screens_data = result["screens"]
    navigation_graph = result["navigation_graph"]

    popups = sum(1 for s in screens_data if s.get("source") == "popup")
    iframes = sum(1 for s in screens_data if s.get("source") == "iframe")
    total_fields = 0
    total_mandatory = 0

    screen_id_by_name: dict[str, str] = {}

    for screen_data in screens_data:
        screen = Screen(
            crawl_run_id=crawl_run.id,
            screen_name=screen_data["screen_name"],
            url=screen_data.get("url"),
            source=screen_data.get("source", "page"),
            parent_screen_id=screen_id_by_name.get(screen_data.get("parent_screen_name")),
        )
        db.add(screen)
        db.flush()
        screen_id_by_name[screen_data["screen_name"]] = screen.id

        entries = build_locator_entries(
            screen_data["elements"],
            screen_name=screen_data["screen_name"],
            transaction_number=transaction_number,
            screen_id=screen.id,
        )
        total_fields += len(entries)
        total_mandatory += sum(1 for e in entries if e["is_mandatory"])

        for entry in entries:
            db.add(LocatorEntry(crawl_run_id=crawl_run.id, **entry))

        # Knowledge capture: the LLM (or rule-based fallback) writes down what
        # it understood about this screen — purpose, field meanings, mandatory
        # set — into the retrievable application memory. Never fatal to a crawl.
        try:
            record_screen_knowledge(db, screen_data["screen_name"], transaction_number, entries)
        except Exception:  # noqa: BLE001
            logger.warning("Knowledge capture failed for screen %s; continuing", screen_data["screen_name"])

    crawl_run.pages_discovered = sum(1 for s in screens_data if s.get("source", "page") == "page")
    crawl_run.popups_discovered = popups
    crawl_run.iframes_discovered = iframes
    crawl_run.fields_discovered = total_fields
    crawl_run.mandatory_fields_discovered = total_mandatory
    crawl_run.navigation_graph = navigation_graph
