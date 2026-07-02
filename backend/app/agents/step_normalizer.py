"""Universal Step Normalizer.

Goal: ANY uploaded feature file executes — regardless of phrasing — against any
screen. Fields already resolve generically at runtime (Locator Repository →
fuzzy match → self-healing), so the only remaining barrier is step WORDING.
This module compiles arbitrary Gherkin phrasing into the canonical Step DSL
(`feature_step_parser`) before execution, in three passes:

  1. CANONICAL  — the line already parses: keep it untouched.
  2. ALIAS      — a library of real-world phrasing patterns (built from the
                  user's production Cucumber wording: "we enter the the account
                  as ...", "click on the ... button", "the user is logged into
                  NBC using ... and ...", "authorize the transaction", etc.)
                  deterministically rewrites the line. No LLM needed.
  3. LLM        — remaining unrecognized lines are sent in ONE batched call to
                  the (optional) Azure OpenAI model with the exact grammar; each
                  suggestion is only accepted if it re-parses canonically.
                  Without an LLM key, unmapped lines are simply reported.

The output is canonical feature text plus a full translation report
(original → canonical → method), so a tester can see exactly how their file
was interpreted before/after a run — nothing is rewritten silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.agents.feature_step_parser import UnrecognizedStep, _parse_step_text

_STEP_LINE = re.compile(r"^(\s*)(Given|When|Then|And|But)\s+(.*?)\s*$", re.IGNORECASE)


@dataclass
class LineMapping:
    original: str
    canonical: str
    method: str  # canonical | alias | llm | unmapped


@dataclass
class NormalizationResult:
    canonical_text: str
    mappings: list[LineMapping] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)

    @property
    def fully_mapped(self) -> bool:
        return not self.unmapped


def _unq(s: str) -> str:
    """Strip optional quotes and articles from a captured field phrase."""
    s = (s or "").strip().strip('"').strip()
    return re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.IGNORECASE).strip()


def _login(u: str, p: str) -> str:
    role = "checker" if "checker" in u.lower() else "maker"
    return f'I am logged in as {role} "{u}" with password "{p}"'


# Ordered alias library: (compiled regex, formatter(match) -> canonical body).
# Order matters — e.g. "clicks on logout button" must hit the logout alias
# before the generic click-button alias swallows it.
_ALIASES: list[tuple[re.Pattern, callable]] = [
    # ---- login ----
    (re.compile(r'^the user is logged in(?:to)?(?: NBC| the application)? using "([^"]+)" and "([^"]+)"$', re.I),
     lambda m: _login(m.group(1), m.group(2))),
    (re.compile(r'^(?:the )?user logs? in(?:to (?:NBC|the application))? (?:with|using) "([^"]+)" and "([^"]+)"$', re.I),
     lambda m: _login(m.group(1), m.group(2))),
    (re.compile(r'^I log ?in as (maker|checker) (?:with|using) "([^"]+)" and "([^"]+)"$', re.I),
     lambda m: f'I am logged in as {m.group(1).lower()} "{m.group(2)}" with password "{m.group(3)}"'),
    # ---- search / open transaction ----
    (re.compile(r'^we search for the transaction "([^"]+)"$', re.I),
     lambda m: f'I search for transaction "{m.group(1)}"'),
    (re.compile(r'^(?:the )?user searches? for (?:the )?transaction "([^"]+)"$', re.I),
     lambda m: f'I search for transaction "{m.group(1)}"'),
    (re.compile(r'^I (?:open|navigate to) (?:the )?transaction "([^"]+)"$', re.I),
     lambda m: f'I search for transaction "{m.group(1)}"'),
    # ---- logout (BEFORE generic click) ----
    (re.compile(r"^(?:the )?(?:user )?clicks? on (?:the )?log ?out(?: button| link)?$", re.I),
     lambda m: "I logout"),
    (re.compile(r"^(?:the )?user logs? ?out(?: of NBC| of the application)?$", re.I),
     lambda m: "I logout"),
    (re.compile(r"^(?:logs? ?out|signs? ?out)$", re.I), lambda m: "I logout"),
    # ---- approve / authorize ----
    (re.compile(r"^(?:the )?(?:user |checker )?authoriz(?:es|e) the transaction\b.*$", re.I),
     lambda m: "I approve the transaction"),
    (re.compile(r"^(?:the )?checker approves(?: the transaction)?\b.*$", re.I),
     lambda m: "I approve the transaction"),
    (re.compile(r"^approves? the transaction\b.+$", re.I),  # with trailing words
     lambda m: "I approve the transaction"),
    # ---- submit for approval ----
    (re.compile(r"^(?:the )?(?:user )?submits? (?:the )?transaction for (?:approval|authorization)$", re.I),
     lambda m: "I submit for approval"),
    (re.compile(r"^(?:I )?submit for authorization$", re.I),
     lambda m: "I submit for approval"),
    # ---- fill ----
    (re.compile(r'^we enter the(?: the)? (.+?) as "([^"]*)"(?: (?:in|on) .*)?$', re.I),
     lambda m: f'I fill "{_unq(m.group(1))}" with "{m.group(2)}"'),
    (re.compile(r'^(?:the )?user enters? "([^"]*)" in(?:to)? (?:the )?(.+?)(?: field)?$', re.I),
     lambda m: f'I fill "{_unq(m.group(2))}" with "{m.group(1)}"'),
    (re.compile(r'^(?:I )?(?:type|enter)s? "([^"]*)" in(?:to)? (?:the )?(.+?)(?: field)?$', re.I),
     lambda m: f'I fill "{_unq(m.group(2))}" with "{m.group(1)}"'),
    (re.compile(r'^set (?:the )?(.+?) to "([^"]*)"$', re.I),
     lambda m: f'I fill "{_unq(m.group(1))}" with "{m.group(2)}"'),
    # ---- select ----
    (re.compile(r'^we select the (.+?) as "([^"]*)"(?: (?:in|on) .*)?$', re.I),
     lambda m: f'I select "{_unq(m.group(1))}" as "{m.group(2)}"'),
    (re.compile(r'^(?:the )?user selects? "([^"]*)" (?:from|in) (?:the )?(.+?)(?: dropdown)?$', re.I),
     lambda m: f'I select "{_unq(m.group(2))}" as "{m.group(1)}"'),
    (re.compile(r'^choose "([^"]*)" (?:from|for) (?:the )?(.+?)$', re.I),
     lambda m: f'I select "{_unq(m.group(2))}" as "{m.group(1)}"'),
    # ---- check ----
    (re.compile(r"^we check the (.+?)(?: checkbox)?$", re.I),
     lambda m: f'I check "{_unq(m.group(1))}"'),
    (re.compile(r"^(?:the )?user (?:checks|ticks) (?:the )?(.+?)(?: checkbox)?$", re.I),
     lambda m: f'I check "{_unq(m.group(1))}"'),
    # ---- click (AFTER logout aliases) ----
    (re.compile(r"^(?:the )?(?:user )?clicks? on (?:the )?(.+?) (?:button|link|tab|icon)$", re.I),
     lambda m: f'I click "{_unq(m.group(1))}"'),
    (re.compile(r'^(?:the )?(?:user )?clicks? on "([^"]+)"$', re.I),
     lambda m: f'I click "{m.group(1)}"'),
    (re.compile(r"^press(?:es)? (?:the )?(.+?)(?: button)?$", re.I),
     lambda m: f'I click "{_unq(m.group(1))}"'),
    # ---- assert text / message (BEFORE assert-visible variants) ----
    (re.compile(r'^I should see (?:the )?(?:message|text) "([^"]+)"$', re.I),
     lambda m: f'I should see text "{m.group(1)}"'),
    (re.compile(r'^"([^"]+)" (?:message )?should be displayed$', re.I),
     lambda m: f'I should see text "{m.group(1)}"'),
    (re.compile(r'^verify (?:the )?(?:message|text) "([^"]+)"$', re.I),
     lambda m: f'I should see text "{m.group(1)}"'),
    # ---- assert field visible ----
    (re.compile(r'^(?:the )?"?(.+?)"? field should be (?:visible|displayed)$', re.I),
     lambda m: f'I should see "{_unq(m.group(1))}"'),
    (re.compile(r'^I should see the "([^"]+)" field$', re.I),
     lambda m: f'I should see "{m.group(1)}"'),
    # ---- wait ----
    (re.compile(r"^(?:I )?waits? (?:for )?(\d+) seconds?$", re.I),
     lambda m: f'I wait "{m.group(1)}" seconds'),
]


def _try_alias(body: str) -> str | None:
    for pattern, formatter in _ALIASES:
        m = pattern.match(body.strip())
        if m:
            canonical = formatter(m)
            # only accept an alias result the real parser actually recognizes
            if not isinstance(_parse_step_text(canonical), UnrecognizedStep):
                return canonical
    return None


def _is_canonical(body: str) -> bool:
    return not isinstance(_parse_step_text(body), UnrecognizedStep)


_LLM_SYSTEM = (
    "You translate Gherkin test steps for a banking app into a strict canonical "
    "grammar. Allowed canonical step bodies (and NOTHING else):\n"
    'I am logged in as maker "<user>" with password "<pass>"\n'
    'I am logged in as checker "<user>" with password "<pass>"\n'
    'I search for transaction "<number or name>"\n'
    'I fill "<field>" with "<value>"\n'
    'I select "<field>" as "<value>"\n'
    'I check "<field>"\n'
    'I click "<button text>"\n'
    "I submit for approval\n"
    "I approve the transaction\n"
    "I logout\n"
    'I should see "<field>"\n'
    'I should see text "<message>"\n'
    'I wait "<N>" seconds\n'
    "Respond ONLY with a JSON array: "
    '[{"original": "<input line>", "canonical": "<canonical body>"}]. '
    'If a line cannot be expressed in this grammar, set "canonical" to null.'
)


def _llm_map(lines: list[str]) -> dict[str, str]:
    from app.llm.azure_openai_client import invoke_llm

    raw = invoke_llm(_LLM_SYSTEM, "Translate these steps:\n" + "\n".join(lines))
    if not raw:
        return {}
    try:
        raw = raw.strip().strip("`")
        raw = raw[raw.index("["):raw.rindex("]") + 1]
        out = {}
        for item in json.loads(raw):
            canonical = item.get("canonical")
            if canonical and _is_canonical(canonical):
                out[item["original"]] = canonical
        return out
    except Exception:
        return {}


def normalize_feature_text(raw_text: str, use_llm: bool = True) -> NormalizationResult:
    lines = raw_text.splitlines()
    parsed: list[dict] = []   # per-line: {out, mapping|None, pending_body|None}
    pending: list[str] = []

    for line in lines:
        m = _STEP_LINE.match(line)
        if not m:
            parsed.append({"out": line, "mapping": None})
            continue
        indent, keyword, body = m.group(1), m.group(2), m.group(3)
        if _is_canonical(body):
            parsed.append({"out": line,
                           "mapping": LineMapping(body, body, "canonical")})
            continue
        alias = _try_alias(body)
        if alias is not None:
            parsed.append({"out": f"{indent}{keyword} {alias}",
                           "mapping": LineMapping(body, alias, "alias")})
            continue
        parsed.append({"out": line, "mapping": None,
                       "pending": (indent, keyword, body)})
        pending.append(body)

    llm_results = _llm_map(pending) if (pending and use_llm) else {}

    result = NormalizationResult(canonical_text="")
    out_lines = []
    for item in parsed:
        if "pending" in item:
            indent, keyword, body = item["pending"]
            canonical = llm_results.get(body)
            if canonical:
                out_lines.append(f"{indent}{keyword} {canonical}")
                result.mappings.append(LineMapping(body, canonical, "llm"))
            else:
                out_lines.append(item["out"])  # left as-is → will fail loudly
                result.mappings.append(LineMapping(body, body, "unmapped"))
                result.unmapped.append(body)
        else:
            out_lines.append(item["out"])
            if item["mapping"]:
                result.mappings.append(item["mapping"])

    result.canonical_text = "\n".join(out_lines) + ("\n" if raw_text.endswith("\n") else "")
    return result
