"""Feature Step DSL Parser.

The Execution Engine needs to deterministically map each Gherkin step onto an action
against the Locator Repository — free-form prose can't be reliably resolved without
an LLM in the loop on every single step (slow, and non-deterministic for a banking
test suite). This module defines a small, explicit Gherkin-shaped step grammar and
parses `.feature` files written in it into typed step objects the Step Executor can
run directly. Lines that don't match a known step are kept as `UnrecognizedStep` so
the engine can fail loudly and informatively rather than silently skip them.
"""

import re
from dataclasses import dataclass

_STEP_LINE = re.compile(r"^\s*(?:Given|When|Then|And|But)\s+(.*)$", re.IGNORECASE)


@dataclass
class LoginStep:
    role: str  # "maker" | "checker"
    username: str
    password: str
    raw: str


@dataclass
class SearchStep:
    query: str
    raw: str


@dataclass
class FillStep:
    field_name: str
    value: str
    raw: str


@dataclass
class SelectStep:
    field_name: str
    value: str
    raw: str


@dataclass
class CheckStep:
    field_name: str
    raw: str


@dataclass
class ClickStep:
    button_text: str
    raw: str


@dataclass
class SubmitForApprovalStep:
    raw: str


@dataclass
class ApproveStep:
    raw: str


@dataclass
class LogoutStep:
    raw: str


@dataclass
class AssertVisibleStep:
    field_name: str
    raw: str


@dataclass
class WaitStep:
    seconds: int
    raw: str


@dataclass
class AssertTextStep:
    text: str
    raw: str


@dataclass
class UnrecognizedStep:
    raw: str


Step = (
    LoginStep
    | SearchStep
    | FillStep
    | SelectStep
    | CheckStep
    | ClickStep
    | SubmitForApprovalStep
    | ApproveStep
    | LogoutStep
    | AssertVisibleStep
    | WaitStep
    | AssertTextStep
    | UnrecognizedStep
)

_PATTERNS: list[tuple[re.Pattern, type]] = [
    (re.compile(r'^I am logged in as (maker|checker) "([^"]*)" with password "([^"]*)"$', re.IGNORECASE), "login"),
    (re.compile(r'^I search for transaction "([^"]*)"$', re.IGNORECASE), "search"),
    (re.compile(r'^I fill "([^"]*)" with "([^"]*)"$', re.IGNORECASE), "fill"),
    (re.compile(r'^I select "([^"]*)" as "([^"]*)"$', re.IGNORECASE), "select"),
    (re.compile(r'^I check "([^"]*)"$', re.IGNORECASE), "check"),
    (re.compile(r'^I click "([^"]*)"$', re.IGNORECASE), "click"),
    (re.compile(r"^I submit for approval$", re.IGNORECASE), "submit_for_approval"),
    (re.compile(r"^I approve the transaction$", re.IGNORECASE), "approve"),
    (re.compile(r"^I logout$", re.IGNORECASE), "logout"),
    (re.compile(r'^I should see text "([^"]*)"$', re.IGNORECASE), "assert_text"),
    (re.compile(r'^I should see "([^"]*)"$', re.IGNORECASE), "assert_visible"),
    (re.compile(r'^I wait "(\d+)" seconds?$', re.IGNORECASE), "wait"),
]


def _parse_step_text(text: str) -> Step:
    for pattern, kind in _PATTERNS:
        match = pattern.match(text.strip())
        if not match:
            continue
        groups = match.groups()
        if kind == "login":
            return LoginStep(role=groups[0].lower(), username=groups[1], password=groups[2], raw=text)
        if kind == "search":
            return SearchStep(query=groups[0], raw=text)
        if kind == "fill":
            return FillStep(field_name=groups[0], value=groups[1], raw=text)
        if kind == "select":
            return SelectStep(field_name=groups[0], value=groups[1], raw=text)
        if kind == "check":
            return CheckStep(field_name=groups[0], raw=text)
        if kind == "click":
            return ClickStep(button_text=groups[0], raw=text)
        if kind == "submit_for_approval":
            return SubmitForApprovalStep(raw=text)
        if kind == "approve":
            return ApproveStep(raw=text)
        if kind == "logout":
            return LogoutStep(raw=text)
        if kind == "assert_visible":
            return AssertVisibleStep(field_name=groups[0], raw=text)
        if kind == "assert_text":
            return AssertTextStep(text=groups[0], raw=text)
        if kind == "wait":
            return WaitStep(seconds=int(groups[0]), raw=text)
    return UnrecognizedStep(raw=text)


def parse_steps(raw_feature_text: str) -> list[Step]:
    steps: list[Step] = []
    for line in raw_feature_text.splitlines():
        match = _STEP_LINE.match(line)
        if not match:
            continue
        steps.append(_parse_step_text(match.group(1)))
    return steps
