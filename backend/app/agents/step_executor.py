"""Step Executor.

Executes a single parsed step (see `feature_step_parser`) against a live Selenium
session: resolves the target field via `locator_resolver` (Locator Repository, with
lightweight self-healing fallback), performs the Selenium action with smart waits,
and reports back what locator was used / whether healing occurred so the caller can
persist an `ExecutionStep` (and `HealingHistory`) row.
"""

from dataclasses import dataclass

import time

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.select import Select
from sqlalchemy.orm import Session

from app.agents.feature_step_parser import (
    ApproveStep,
    AssertTextStep,
    AssertVisibleStep,
    CheckStep,
    ClickStep,
    FillStep,
    LoginStep,
    LogoutStep,
    SearchStep,
    SelectStep,
    Step,
    SubmitForApprovalStep,
    UnrecognizedStep,
    WaitStep,
)
from app.agents.locator_resolver import ElementResolutionError, resolve_element
from app.agents.login_agent import perform_login
from app.agents.smart_wait import wait_until_clickable, wait_for_document_ready
from app.agents.transaction_search_agent import search_transaction

_LOGOUT_HINTS = ["logout", "log out", "sign out"]


@dataclass
class StepOutcome:
    status: str  # "passed" | "failed"
    locator_used: str | None = None
    locator_type: str | None = None
    healed: bool = False
    healing_info: dict | None = None
    error: Exception | None = None


def _find_button_by_text(driver: WebDriver, text: str) -> WebElement | None:
    needle = text.strip().lower()
    elements = driver.find_elements(By.CSS_SELECTOR, "button, input[type='submit'], input[type='button'], a")
    for el in elements:
        if not el.is_displayed():
            continue
        candidate = (el.text or el.get_attribute("value") or "").strip().lower()
        if needle in candidate or candidate in needle:
            return el
    return None


def execute_step(driver: WebDriver, db: Session, step: Step, context: dict) -> StepOutcome:
    try:
        # Every step starts from the top-level document. `resolve_element` switches
        # into an iframe when a field lives there and leaves the driver positioned
        # inside it; without this reset, the next step (e.g. clicking "Next" in the
        # parent document) would be searched for inside that stale iframe context.
        driver.switch_to.default_content()

        if isinstance(step, LoginStep):
            wait_for_document_ready(driver)
            perform_login(driver, driver.current_url, step.username, step.password)
            context["role"] = step.role
            return StepOutcome("passed")

        if isinstance(step, SearchStep):
            search_transaction(driver, step.query, None)
            context["transaction_number"] = step.query
            return StepOutcome("passed")

        if isinstance(step, FillStep):
            resolved = resolve_element(driver, db, context.get("transaction_number"), step.field_name)
            wait_until_clickable(resolved.element, driver)
            resolved.element.clear()
            resolved.element.send_keys(step.value)
            return StepOutcome("passed", resolved.locator_used, resolved.locator_type, resolved.healed, resolved.healing_info)

        if isinstance(step, SelectStep):
            # `field_name` is a <select> dropdown's own label in our locator builder,
            # but for radio-button-style groups it's just the group's caption text —
            # there is no element for "Transfer Mode" itself, only for each option
            # ("NEFT", "RTGS", ...). Try the dropdown path first; fall back to
            # resolving the option value directly when the field name itself doesn't
            # resolve to anything.
            try:
                resolved = resolve_element(driver, db, context.get("transaction_number"), step.field_name)
            except ElementResolutionError:
                resolved = None

            if resolved is not None and resolved.element.tag_name.lower() == "select":
                wait_until_clickable(resolved.element, driver)
                Select(resolved.element).select_by_visible_text(step.value)
                return StepOutcome("passed", resolved.locator_used, resolved.locator_type, resolved.healed, resolved.healing_info)

            option = resolve_element(driver, db, context.get("transaction_number"), step.value)
            wait_until_clickable(option.element, driver)
            option.element.click()
            return StepOutcome("passed", option.locator_used, option.locator_type, option.healed, option.healing_info)

        if isinstance(step, CheckStep):
            resolved = resolve_element(driver, db, context.get("transaction_number"), step.field_name)
            wait_until_clickable(resolved.element, driver)
            if not resolved.element.is_selected():
                resolved.element.click()
            return StepOutcome("passed", resolved.locator_used, resolved.locator_type, resolved.healed, resolved.healing_info)

        if isinstance(step, ClickStep):
            element = _find_button_by_text(driver, step.button_text)
            if element is None:
                raise ElementResolutionError(step.button_text, "no button/link matched this text")
            wait_until_clickable(element, driver)
            element.click()
            wait_for_document_ready(driver)
            return StepOutcome("passed", locator_used=f"text={step.button_text}", locator_type="text")

        if isinstance(step, SubmitForApprovalStep):
            element = _find_button_by_text(driver, "submit for approval") or _find_button_by_text(driver, "submit")
            if element is None:
                raise ElementResolutionError("Submit for Approval", "no matching button found")
            wait_until_clickable(element, driver)
            element.click()
            # This button submits a real form POST + redirect — without waiting for
            # the resulting navigation, a `driver.quit()` immediately after the last
            # step in a feature file can race ahead of the request actually landing.
            wait_for_document_ready(driver)
            return StepOutcome("passed", locator_used="text=Submit for Approval", locator_type="text")

        if isinstance(step, ApproveStep):
            element = _find_button_by_text(driver, "approve")
            if element is None:
                raise ElementResolutionError("Approve", "no matching button found")
            wait_until_clickable(element, driver)
            element.click()
            wait_for_document_ready(driver)
            return StepOutcome("passed", locator_used="text=Approve", locator_type="text")

        if isinstance(step, LogoutStep):
            element = None
            for hint in _LOGOUT_HINTS:
                element = _find_button_by_text(driver, hint)
                if element:
                    break
            if element is not None:
                element.click()
            else:
                driver.delete_all_cookies()
            return StepOutcome("passed")

        if isinstance(step, AssertVisibleStep):
            resolved = resolve_element(driver, db, context.get("transaction_number"), step.field_name)
            if not resolved.element.is_displayed():
                raise AssertionError(f"Field '{step.field_name}' was found but is not visible")
            return StepOutcome("passed", resolved.locator_used, resolved.locator_type, resolved.healed, resolved.healing_info)

        if isinstance(step, WaitStep):
            # explicit tester-requested pause, capped defensively
            time.sleep(min(step.seconds, 60))
            wait_for_document_ready(driver)
            return StepOutcome("passed")

        if isinstance(step, AssertTextStep):
            # visible-text assertion: top-level document first, then one level of
            # iframes (matching the crawler's/resolver's iframe depth)
            needle = step.text.strip().lower()

            def _visible_text() -> str:
                try:
                    return (driver.find_element(By.TAG_NAME, "body").text or "").lower()
                except Exception:  # noqa: BLE001
                    return ""

            found = needle in _visible_text()
            if not found:
                frames = driver.find_elements(By.CSS_SELECTOR, "iframe, frame")
                for index in range(len(frames)):
                    try:
                        driver.switch_to.default_content()
                        driver.switch_to.frame(index)
                        if needle in _visible_text():
                            found = True
                            break
                    except Exception:  # noqa: BLE001
                        continue
                driver.switch_to.default_content()
            if not found:
                raise AssertionError(f"Expected text not visible on screen: '{step.text}'")
            return StepOutcome("passed")

        if isinstance(step, UnrecognizedStep):
            raise ValueError(f"Unrecognized step: {step.raw}")

        raise ValueError(f"No executor implemented for step type {type(step).__name__}")
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure becomes a classified step failure
        return StepOutcome("failed", error=exc)
