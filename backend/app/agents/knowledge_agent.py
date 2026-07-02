"""Application Knowledge Agent — the platform's "learning" layer.

The LLM observes what the crawler discovers and what executions do, writes down
what it understood, and that memory is retrieved later to make feature-file
generation and failure repair smarter. This is deliberately RAG (retrieve-and-
ground), NOT fine-tuning: memory updates the moment something is learned, costs
one small LLM call per screen/run, and — following this codebase's rule — every
capture path has a rule-based fallback so the platform never depends on an LLM
key being configured.

What gets captured:
  during a CRAWL      → one "screen" item per discovered screen: purpose,
                        field meanings, mandatory set, navigation position.
  during an EXECUTION → one "lesson" item per self-heal ("field X's locator
                        broke, new one is Y") and per failure ("clicking Submit
                        on screen Z fails with ... — root cause ..."), plus one
                        "episode" item summarizing the whole run.
  retrieval           → `retrieve()` semantically queries Chroma; the feature
                        generation graph injects the top matches into its
                        drafting/repair prompts.

`build_finetuning_records()` additionally converts accumulated knowledge into
chat-format JSONL, so if a real fine-tune is ever justified, the dataset has
been accumulating from day one.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.llm.azure_openai_client import invoke_llm
from app.models.knowledge import KnowledgeItem

logger = logging.getLogger(__name__)

_MAX_ITEMS_PER_QUERY = 5


# --------------------------------------------------------------------------
# Capture: screens (called from crawl_service after a crawl persists)
# --------------------------------------------------------------------------

def summarize_screen(screen_name: str, transaction_number: str | None,
                     fields: list[dict]) -> tuple[str, str]:
    """Returns (content, source). Rule-based summary always works; the LLM
    upgrades it into a business-language explanation when configured."""
    mandatory = [f["field_name"] for f in fields if f.get("is_mandatory")]
    optional = [f["field_name"] for f in fields if not f.get("is_mandatory")]
    dropdowns = {f["field_name"]: f.get("dropdown_options")
                 for f in fields if f.get("dropdown_options")}
    rule_summary = (
        f"Screen '{screen_name}'"
        + (f" (transaction {transaction_number})" if transaction_number else "")
        + f" has {len(fields)} fields. Mandatory: {mandatory or 'none'}. "
        + f"Optional: {optional or 'none'}."
        + (f" Dropdowns: {dropdowns}." if dropdowns else "")
    )

    llm_summary = invoke_llm(
        "You are analyzing a banking application screen for a test-automation "
        "knowledge base. In 3-5 sentences, explain: the screen's likely business "
        "purpose, what each field means in banking terms, which fields look "
        "interdependent, and anything a test author should know (e.g. fields that "
        "probably require valid existing data). Be concrete, no filler.",
        rule_summary + "\nField details: "
        + json.dumps([{k: f.get(k) for k in ("field_name", "control_type",
                                             "is_mandatory", "dropdown_options")}
                      for f in fields])[:6000],
    )
    if llm_summary:
        return rule_summary + "\nLLM analysis: " + llm_summary.strip(), "llm"
    return rule_summary, "rule"


def record_screen_knowledge(db: Session, screen_name: str,
                            transaction_number: str | None,
                            fields: list[dict]) -> KnowledgeItem:
    content, source = summarize_screen(screen_name, transaction_number, fields)
    item = KnowledgeItem(
        kind="screen", transaction_number=transaction_number,
        screen_name=screen_name,
        title=f"Screen understanding: {screen_name}",
        content=content, source=source,
        meta={"field_count": len(fields)},
    )
    db.add(item)
    db.flush()
    _index(item)
    return item


# --------------------------------------------------------------------------
# Capture: lessons + episodes (called from execution_service)
# --------------------------------------------------------------------------

def record_healing_lesson(db: Session, transaction_number: str | None,
                          field_name: str, old_locator: str | None,
                          new_locator: str, explanation: str | None) -> KnowledgeItem:
    content = (
        f"Self-healing on field '{field_name}'"
        + (f" (transaction {transaction_number})" if transaction_number else "")
        + f": stored locator '{old_locator}' no longer resolved; "
        + f"replaced with '{new_locator}'."
        + (f" Explanation: {explanation}" if explanation else "")
    )
    item = KnowledgeItem(
        kind="lesson", transaction_number=transaction_number,
        title=f"Locator healed: {field_name}",
        content=content, source="llm" if explanation else "rule",
        meta={"field_name": field_name, "old": old_locator, "new": new_locator},
    )
    db.add(item)
    db.flush()
    _index(item)
    return item


def record_failure_lesson(db: Session, transaction_number: str | None,
                          step_text: str, category: str | None,
                          root_cause: str | None,
                          suggested_fix: str | None) -> KnowledgeItem:
    content = (
        f"Step failed: {step_text}"
        + (f" (transaction {transaction_number})" if transaction_number else "")
        + f". Category: {category or 'unknown'}."
        + (f" Root cause: {root_cause}." if root_cause else "")
        + (f" Suggested fix: {suggested_fix}." if suggested_fix else "")
    )
    item = KnowledgeItem(
        kind="lesson", transaction_number=transaction_number,
        title=f"Failure lesson: {category or 'unknown'}",
        content=content, source="rule",
        meta={"step_text": step_text, "category": category},
    )
    db.add(item)
    db.flush()
    _index(item)
    return item


def summarize_execution(transaction_number: str | None, step_rows: list[dict],
                        status: str) -> tuple[str, str]:
    passed = sum(1 for s in step_rows if s.get("status") == "passed")
    failed = [s for s in step_rows if s.get("status") == "failed"]
    healed = [s for s in step_rows if s.get("healed")]
    rule_summary = (
        f"Execution {status}"
        + (f" for transaction {transaction_number}" if transaction_number else "")
        + f": {passed}/{len(step_rows)} steps passed, {len(healed)} healed."
        + (f" Failed steps: {[s.get('step_text') for s in failed]}."
           if failed else "")
    )
    llm_summary = invoke_llm(
        "You observe banking test-automation runs. In 2-4 sentences describe what "
        "this run actually did in business terms (what workflow, what data), what "
        "went wrong if anything, and one lesson worth remembering for future runs "
        "of this transaction. No filler.",
        rule_summary + "\nSteps: "
        + json.dumps([{k: s.get(k) for k in ("step_text", "status", "healed",
                                             "error_message")}
                      for s in step_rows])[:6000],
    )
    if llm_summary:
        return rule_summary + "\nLLM analysis: " + llm_summary.strip(), "llm"
    return rule_summary, "rule"


def record_execution_episode(db: Session, transaction_number: str | None,
                             step_rows: list[dict], status: str) -> KnowledgeItem:
    content, source = summarize_execution(transaction_number, step_rows, status)
    item = KnowledgeItem(
        kind="episode", transaction_number=transaction_number,
        title=f"Execution episode ({status})",
        content=content, source=source,
        meta={"step_count": len(step_rows), "status": status},
    )
    db.add(item)
    db.flush()
    _index(item)
    return item


# --------------------------------------------------------------------------
# Retrieval — used by the feature generation graph and available via API
# --------------------------------------------------------------------------

def retrieve(query: str, n: int = _MAX_ITEMS_PER_QUERY) -> list[dict]:
    try:
        from app.vectorstore.chroma_client import query_knowledge  # lazy: optional dep
        return query_knowledge(query, n_results=n)
    except Exception:  # noqa: BLE001
        logger.warning("Knowledge retrieval unavailable; returning no matches")
        return []


def knowledge_context_for(transaction_number: str, intent: str = "") -> str:
    """Formatted memory block for injection into generation/repair prompts."""
    hits = retrieve(f"transaction {transaction_number} {intent}".strip())
    if not hits:
        return ""
    lines = [f"- {h['text']}" for h in hits]
    return ("Relevant knowledge from previous crawls and runs of this "
            "application:\n" + "\n".join(lines))


def _index(item: KnowledgeItem) -> None:
    try:
        from app.vectorstore.chroma_client import add_knowledge_items  # lazy: optional dep
    except Exception:  # noqa: BLE001
        logger.warning("Vector index unavailable; knowledge stored in DB only")
        return
    add_knowledge_items(
        [item.id], [item.content],
        [{"kind": item.kind, "transaction_number": item.transaction_number or "",
          "screen_name": item.screen_name or "", "title": item.title}],
    )


# --------------------------------------------------------------------------
# Fine-tuning export — accumulate now, fine-tune later (if ever justified)
# --------------------------------------------------------------------------

def build_finetuning_records(items: list[KnowledgeItem]) -> list[dict]:
    """Chat-format records (one JSON object per line when written as JSONL),
    compatible with Azure OpenAI fine-tuning. Screens become 'explain this
    screen' pairs; lessons become 'diagnose this failure' pairs; episodes
    become 'summarize this run' pairs."""
    prompts = {
        "screen": "Explain this banking application screen to a test author:",
        "lesson": "Given this automation failure/heal, state the lesson learned:",
        "episode": "Summarize what this automation run did and what to remember:",
    }
    records = []
    for item in items:
        records.append({
            "messages": [
                {"role": "system",
                 "content": "You are an NBC banking application automation expert."},
                {"role": "user",
                 "content": f"{prompts.get(item.kind, prompts['episode'])} {item.title}"},
                {"role": "assistant", "content": item.content},
            ]
        })
    return records


def export_finetuning_jsonl(db: Session, path) -> int:
    items = db.query(KnowledgeItem).order_by(KnowledgeItem.created_at).all()
    records = build_finetuning_records(items)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)
