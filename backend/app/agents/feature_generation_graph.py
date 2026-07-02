"""Feature Generation Graph (LangGraph).

Turns (transaction number + plain-English intent + optional field values) into a
validated — and optionally executed and self-repaired — feature file in this
platform's Step DSL, grounded ONLY in the Locator Repository so the LLM cannot
hallucinate field names.

Graph topology:

    load_repository ──► draft_feature ──► validate_draft ──┐
          │ (empty repo → finalize)          ▲             │ missing fields &
          ▼                                  │             │ attempts left
       finalize ◄──────────────┐        repair_draft ◄─────┘
          ▲                    │             (fuzzy-map / LLM rewrite)
          │ passed / no exec   │
    execute_run ◄── valid ─────┘
          │ failed & attempts left
          ▼
    repair_after_run ──► execute_run (retry)   [app_error/assertion → finalize]

Design rules (matching this codebase's conventions):
- LLM is OPTIONAL. With no Azure OpenAI key configured, drafting falls back to a
  deterministic builder (mandatory repo fields + caller-supplied field values,
  fuzzy-matched to repository field names) and repair falls back to fuzzy renaming.
- Validation reuses `locator_validation_agent.validate` (the same pre-execution
  check the Execution Center uses).
- Execution reuses `execution_orchestrator.run_execution` unchanged, with
  in-memory callbacks; failures are classified via `failure_analysis_agent`.
- Heavy imports (selenium et al.) are deferred into the nodes that need them so
  draft-only usage never touches a browser stack.

Requires `langgraph` (added to requirements.txt). The import is lazy with a clear
error message so the rest of the app keeps working if it isn't installed yet.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Optional, TypedDict

MAX_DRAFT_ATTEMPTS = 3
MAX_RUN_ATTEMPTS = 2

_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "field", "no", "number"}


# --------------------------------------------------------------------------
# Pure helpers (unit-testable without DB/browser/LLM)
# --------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return (name or "").strip().rstrip("*").strip().lower()


def _words(name: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", _normalize(name)) if w and w not in _STOPWORDS}


def fuzzy_best_match(phrase: str, candidates: list[str]) -> Optional[str]:
    """Word-overlap first, character similarity only as a last resort.

    Rationale (mirrors the Phase-4 bug described in the README): raw character
    similarity scores 'Amount' above 'Account Number' for the phrase 'the
    account', which is backwards. Significant-word overlap gets it right.
    """
    if not candidates:
        return None
    p_words = _words(phrase)
    best, best_score = None, 0.0
    for cand in candidates:
        overlap = len(p_words & _words(cand))
        if overlap > 0:
            # prefer more overlapping words; tie-break on shorter candidate
            score = overlap + 1.0 / (1 + len(cand))
            if score > best_score:
                best, best_score = cand, score
    if best is not None:
        return best
    # no word overlap anywhere → character-level fallback
    for cand in candidates:
        score = SequenceMatcher(None, _normalize(phrase), _normalize(cand)).ratio()
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= 0.55 else None


def build_deterministic_draft(
    transaction_number: str,
    repo_fields: list[dict],
    field_values: dict[str, str],
    needs_approval: bool,
    maker_username: str,
    maker_password: str,
    checker_username: str = "",
    checker_password: str = "",
) -> str:
    """No-LLM draft: fill every caller-supplied value (fuzzy-mapped onto real repo
    field names) plus any mandatory repo field the caller didn't cover (with an
    obvious placeholder the human can review), then submit / approve."""
    names = [f["field_name"] for f in repo_fields]
    by_name = {f["field_name"]: f for f in repo_fields}

    lines = [
        f"Feature: Auto-generated workflow for transaction {transaction_number}",
        "",
        f"  Scenario: Execute transaction {transaction_number}",
        f'    Given I am logged in as maker "{maker_username}" with password "{maker_password}"',
        f'    When I search for transaction "{transaction_number}"',
    ]

    covered: set[str] = set()
    for phrase, value in (field_values or {}).items():
        target = fuzzy_best_match(phrase, names) or phrase
        covered.add(_normalize(target))
        entry = by_name.get(target, {})
        if entry.get("control_type") == "select" or entry.get("dropdown_options"):
            lines.append(f'    And I select "{target}" as "{value}"')
        elif entry.get("control_type") == "checkbox":
            lines.append(f'    And I check "{target}"')
        else:
            lines.append(f'    And I fill "{target}" with "{value}"')

    for f in repo_fields:
        if f.get("is_mandatory") and _normalize(f["field_name"]) not in covered:
            if f.get("control_type") == "select" and f.get("dropdown_options"):
                lines.append(f'    And I select "{f["field_name"]}" as "{f["dropdown_options"][0]}"')
            else:
                lines.append(f'    And I fill "{f["field_name"]}" with "TODO_VALUE"')

    if needs_approval:
        lines.append("    And I submit for approval")
        lines.append("    And I logout")
        lines.append(
            f'    Given I am logged in as checker "{checker_username}" with password "{checker_password}"'
        )
        lines.append(f'    When I search for transaction "{transaction_number}"')
        lines.append("    And I approve the transaction")
    else:
        lines.append('    And I click "Submit"')
    lines.append("    And I logout")
    return "\n".join(lines) + "\n"


def rewrite_field_names(draft: str, renames: dict[str, str]) -> str:
    """Replace quoted field names in Fill/Select/Check/Should-see lines only."""
    out = draft
    for old, new in renames.items():
        out = re.sub(
            r'((?:fill|select|check|should see)\s+")' + re.escape(old) + '(")',
            lambda m: m.group(1) + new + m.group(2),
            out,
            flags=re.IGNORECASE,
        )
    return out


def extract_gherkin(text: str) -> str:
    """LLM responses sometimes wrap output in ``` fences — strip them."""
    m = re.search(r"```(?:gherkin)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------

class GenState(TypedDict, total=False):
    transaction_number: str
    intent: str
    field_values: dict
    needs_approval: bool
    execute: bool
    base_url: str
    headless: Optional[bool]
    maker_username: str
    maker_password: str
    checker_username: str
    checker_password: str

    repo_fields: list          # [{field_name, control_type, is_mandatory, dropdown_options, confidence}]
    draft: str
    validation: dict           # asdict(ValidationReport)
    draft_attempts: int
    run_attempts: int
    step_results: list         # per-step dicts from the last run
    status: str                # drafted | passed | failed | no_repository | error
    notes: list                # human-readable audit trail of what the graph did
    root_cause: Optional[str]


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------

class FeatureGenerationGraph:
    """Holds the DB session and compiles the LangGraph once per instance."""

    def __init__(self, db):
        self.db = db
        try:
            from langgraph.graph import StateGraph, END  # lazy: clear error if absent
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "langgraph is not installed — run `pip install langgraph` "
                "(it is listed in requirements.txt)."
            ) from exc
        self._END = END
        g = StateGraph(GenState)
        g.add_node("load_repository", self.load_repository)
        g.add_node("draft_feature", self.draft_feature)
        g.add_node("validate_draft", self.validate_draft)
        g.add_node("repair_draft", self.repair_draft)
        g.add_node("execute_run", self.execute_run)
        g.add_node("repair_after_run", self.repair_after_run)
        g.add_node("finalize", self.finalize)

        g.set_entry_point("load_repository")
        g.add_conditional_edges(
            "load_repository",
            lambda s: "finalize" if s.get("status") == "no_repository" else "draft_feature",
            {"finalize": "finalize", "draft_feature": "draft_feature"},
        )
        g.add_edge("draft_feature", "validate_draft")
        g.add_conditional_edges("validate_draft", self._after_validate,
                                {"repair_draft": "repair_draft",
                                 "execute_run": "execute_run",
                                 "finalize": "finalize"})
        g.add_edge("repair_draft", "validate_draft")
        g.add_conditional_edges("execute_run", self._after_run,
                                {"repair_after_run": "repair_after_run",
                                 "finalize": "finalize"})
        g.add_conditional_edges("repair_after_run",
                                lambda s: "finalize" if s.get("status") == "failed" else "execute_run",
                                {"execute_run": "execute_run", "finalize": "finalize"})
        g.add_edge("finalize", END)
        self.graph = g.compile()

    # ---------------- nodes ----------------

    def load_repository(self, state: GenState) -> GenState:
        from app.agents.locator_resolver import effective_entries_for_transaction

        entries = effective_entries_for_transaction(self.db, state["transaction_number"])
        # keep only real, named form fields
        fields = []
        seen = set()
        for e in entries:
            name = (e.field_name or "").strip()
            if not name or _normalize(name) in seen:
                continue
            seen.add(_normalize(name))
            fields.append({
                "field_name": name,
                "control_type": e.control_type,
                "is_mandatory": bool(e.is_mandatory),
                "dropdown_options": e.dropdown_options or [],
                "confidence": e.ai_confidence_score or 0.0,
            })
        state["repo_fields"] = fields
        state["notes"] = state.get("notes", []) + [f"repository: {len(fields)} distinct fields"]
        if not fields:
            state["status"] = "no_repository"
            state["root_cause"] = (
                f"No Locator Repository entries for transaction "
                f"'{state['transaction_number']}'. Crawl it first (Module 1) or "
                f"upload a locator file (Module 3)."
            )
        return state

    def draft_feature(self, state: GenState) -> GenState:
        state["draft_attempts"] = state.get("draft_attempts", 0) + 1
        maker_u = state.get("maker_username") or os.environ.get("MAKER_USERNAME", "")
        maker_p = state.get("maker_password") or os.environ.get("MAKER_PASSWORD", "")
        checker_u = state.get("checker_username") or os.environ.get("CHECKER_USERNAME", "")
        checker_p = state.get("checker_password") or os.environ.get("CHECKER_PASSWORD", "")

        llm_draft = self._llm_draft(state, maker_u, maker_p, checker_u, checker_p)
        if llm_draft:
            state["draft"] = llm_draft
            state["notes"].append("draft: LLM-generated (repo-grounded)")
        else:
            state["draft"] = build_deterministic_draft(
                state["transaction_number"], state["repo_fields"],
                state.get("field_values") or {}, bool(state.get("needs_approval")),
                maker_u, maker_p, checker_u, checker_p,
            )
            state["notes"].append("draft: deterministic builder (no LLM configured)")
        return state

    def validate_draft(self, state: GenState) -> GenState:
        from app.agents.locator_validation_agent import validate

        report = validate(self.db, state["transaction_number"], [state["draft"]])
        state["validation"] = asdict(report)
        state["notes"].append(
            f"validation: {report.mapped_count}/{report.total_steps} field steps mapped, "
            f"missing={report.missing_fields}"
        )
        return state

    def _after_validate(self, state: GenState) -> str:
        missing = state["validation"].get("missing_fields") or []
        if missing and state["draft_attempts"] < MAX_DRAFT_ATTEMPTS:
            return "repair_draft"
        if missing:
            state["status"] = "failed"
            state["root_cause"] = f"Unresolvable field names after repair attempts: {missing}"
            return "finalize"
        if state.get("execute"):
            return "execute_run"
        state["status"] = "drafted"
        return "finalize"

    def repair_draft(self, state: GenState) -> GenState:
        state["draft_attempts"] = state.get("draft_attempts", 0) + 1
        names = [f["field_name"] for f in state["repo_fields"]]
        renames = {}
        for missing in state["validation"].get("missing_fields", []):
            match = fuzzy_best_match(missing, names)
            if match:
                renames[missing] = match
        if renames:
            state["draft"] = rewrite_field_names(state["draft"], renames)
            state["notes"].append(f"repair: fuzzy-renamed {renames}")
            return state
        # fuzzy couldn't fix it — one LLM shot at rewriting, else drop the lines
        fixed = self._llm_repair(state)
        if fixed:
            state["draft"] = fixed
            state["notes"].append("repair: LLM rewrote draft against repository field list")
        else:
            missing = set(_normalize(m) for m in state["validation"].get("missing_fields", []))
            kept = [ln for ln in state["draft"].splitlines()
                    if not any(_normalize(m) in _normalize(ln) for m in missing)
                    or not re.search(r'\b(fill|select|check|should see)\b', ln, re.I)]
            state["draft"] = "\n".join(kept) + "\n"
            state["notes"].append(f"repair: dropped steps for unmatchable fields {sorted(missing)}")
        return state

    def execute_run(self, state: GenState) -> GenState:
        from pathlib import Path
        from app.agents.execution_orchestrator import run_execution
        from app.agents.failure_analysis_agent import classify_failure
        from app.core.config import get_settings

        state["run_attempts"] = state.get("run_attempts", 0) + 1
        settings = get_settings()
        shots = Path(settings.export_dir) / "screenshots" / "generation_dry_runs"
        step_results: list[dict] = []

        def on_step(result: dict) -> None:
            rec = {
                "sequence": result["sequence"],
                "step_text": getattr(result["step"], "raw", str(result["step"])),
                "step_type": type(result["step"]).__name__,
                "status": result["status"],
                "healed": result["healed"],
                "duration_ms": result["duration_ms"],
                "error": str(result["error"]) if result["error"] else None,
                "failure_category": None,
            }
            if result["error"] is not None:
                analysis = classify_failure(result["step"], result["error"])
                rec["failure_category"] = getattr(analysis, "category", None)
                rec["suggested_fix"] = getattr(analysis, "suggested_fix", None)
            step_results.append(rec)

        totals = run_execution(
            base_url=state["base_url"],
            feature_files=[{"filename": f"generated_{state['transaction_number']}.feature",
                            "raw_text": state["draft"]}],
            failure_mode="stop",
            db=self.db,
            screenshot_dir=shots,
            on_feature_file_status=lambda *_: None,
            on_step_complete=on_step,
            headless=state.get("headless"),
        )
        state["step_results"] = step_results
        state["notes"].append(
            f"run #{state['run_attempts']}: {totals['passed']}/{totals['total']} passed, "
            f"{totals['healed']} healed"
        )
        return state

    def _after_run(self, state: GenState) -> str:
        failed = [s for s in state.get("step_results", []) if s["status"] == "failed"]
        if not failed:
            state["status"] = "passed"
            return "finalize"
        if state["run_attempts"] >= MAX_RUN_ATTEMPTS:
            state["status"] = "failed"
            state["root_cause"] = failed[0].get("error")
            return "finalize"
        return "repair_after_run"

    def repair_after_run(self, state: GenState) -> GenState:
        failed = [s for s in state["step_results"] if s["status"] == "failed"][0]
        category = failed.get("failure_category")
        state["notes"].append(
            f"diagnosing failed step '{failed['step_text']}' (category={category})"
        )
        if category in ("assertion", "app_error"):
            # a real application/validation failure — retrying identical steps is
            # pointless and, on a banking system, potentially harmful
            state["status"] = "failed"
            state["root_cause"] = (
                f"Application-level failure, not an automation defect: "
                f"{failed.get('error')} — {failed.get('suggested_fix') or ''}"
            )
            return state
        if category == "locator_not_found":
            m = re.search(r'"([^"]+)"', failed["step_text"])
            if m:
                names = [f["field_name"] for f in state["repo_fields"]
                         if _normalize(f["field_name"]) != _normalize(m.group(1))]
                match = fuzzy_best_match(m.group(1), names)
                if match:
                    state["draft"] = rewrite_field_names(state["draft"], {m.group(1): match})
                    state["notes"].append(f"repair: '{m.group(1)}' → '{match}', retrying")
                    return state
            fixed = self._llm_repair(state, failed)
            if fixed:
                state["draft"] = fixed
                state["notes"].append("repair: LLM rewrote draft after run failure, retrying")
                return state
            state["status"] = "failed"
            state["root_cause"] = failed.get("error")
            return state
        # timeout / undefined_step / unknown → self-healing + smart waits already
        # ran inside the executor; one plain retry covers transient cases
        state["notes"].append("repair: transient category, retrying as-is")
        return state

    def finalize(self, state: GenState) -> GenState:
        from app.services.feature_file_service import ingest_feature_file

        if state.get("draft") and state.get("status") in ("drafted", "passed"):
            doc = ingest_feature_file(
                self.db,
                filename=f"generated_txn_{state['transaction_number']}.feature",
                raw_text=state["draft"],
            )
            self.db.commit()
            state["notes"].append(f"saved feature file doc id={doc.id}")
        return state

    # ---------------- LLM helpers (always optional) ----------------

    def _llm_draft(self, state, maker_u, maker_p, checker_u, checker_p) -> Optional[str]:
        from app.llm.azure_openai_client import invoke_llm

        fields_desc = "\n".join(
            f"- {f['field_name']} (type={f['control_type']}, mandatory={f['is_mandatory']}"
            + (f", options={f['dropdown_options']}" if f["dropdown_options"] else "") + ")"
            for f in state["repo_fields"]
        )
        approval = "The transaction REQUIRES checker approval." if state.get("needs_approval") \
            else "No checker approval is needed."
        system = (
            "You generate Gherkin feature files for a banking automation platform. "
            "You MUST use ONLY these exact step patterns:\n"
            'Given I am logged in as maker "<user>" with password "<pass>"\n'
            'Given I am logged in as checker "<user>" with password "<pass>"\n'
            'When I search for transaction "<number>"\n'
            'And I fill "<field>" with "<value>"\n'
            'And I select "<field>" as "<value>"\n'
            'And I check "<field>"\n'
            'And I click "<button text>"\n'
            "And I submit for approval\nAnd I approve the transaction\nAnd I logout\n"
            'And I should see "<field>"\n'
            "Field names MUST be copied character-for-character from the provided field "
            "list — never invent, abbreviate, or rephrase a field name. Fill every "
            "mandatory field. Output ONLY the feature file, no explanation."
        )
        try:
            from app.agents.knowledge_agent import knowledge_context_for
            memory = knowledge_context_for(state["transaction_number"], state.get("intent", ""))
        except Exception:  # noqa: BLE001
            memory = ""
        user = (
            (memory + "\n\n" if memory else "")
            + f"Transaction number: {state['transaction_number']}\n"
            f"Tester's intent: {state.get('intent') or 'execute this transaction with valid data'}\n"
            f"Known values to use (map them onto the closest real field names): "
            f"{state.get('field_values') or {}}\n{approval}\n"
            f"Maker credentials: {maker_u} / {maker_p}\n"
            f"Checker credentials: {checker_u} / {checker_p}\n"
            f"Available fields from the Locator Repository:\n{fields_desc}"
        )
        raw = invoke_llm(system, user)
        return extract_gherkin(raw) if raw else None

    def _llm_repair(self, state, failed_step: dict | None = None) -> Optional[str]:
        from app.llm.azure_openai_client import invoke_llm

        names = ", ".join(f'"{f["field_name"]}"' for f in state["repo_fields"])
        problem = (
            f"This step failed at runtime: {failed_step['step_text']} "
            f"({failed_step.get('error')})" if failed_step
            else f"Validation reported missing fields: {state['validation'].get('missing_fields')}"
        )
        raw = invoke_llm(
            "You fix Gherkin feature files. Keep every step pattern identical; only "
            "correct field names so each one exactly matches a name from the allowed "
            "list, or remove a step whose field has no plausible match. Output ONLY "
            "the corrected feature file.",
            f"Allowed field names: {names}\n{problem}\nFeature file:\n{state['draft']}",
        )
        return extract_gherkin(raw) if raw else None

    # ---------------- public API ----------------

    def run(self, **kwargs) -> dict:
        initial: GenState = {
            "draft_attempts": 0, "run_attempts": 0, "notes": [],
            "status": "error", **kwargs,
        }
        final = self.graph.invoke(initial)
        return {
            "status": final.get("status"),
            "feature_text": final.get("draft"),
            "validation": final.get("validation"),
            "step_results": final.get("step_results"),
            "notes": final.get("notes"),
            "root_cause": final.get("root_cause"),
            "repo_field_count": len(final.get("repo_fields") or []),
        }
