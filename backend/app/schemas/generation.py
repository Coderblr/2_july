from typing import Optional

from pydantic import BaseModel


class GenerationRequest(BaseModel):
    transaction_number: str
    intent: str = ""
    field_values: dict[str, str] = {}
    needs_approval: bool = False
    # Credentials: optional — fall back to MAKER_USERNAME / MAKER_PASSWORD /
    # CHECKER_USERNAME / CHECKER_PASSWORD environment variables when omitted.
    maker_username: str = ""
    maker_password: str = ""
    checker_username: str = ""
    checker_password: str = ""


class GenerateAndRunRequest(GenerationRequest):
    base_url: str
    headless: Optional[bool] = None


class GenerationResponse(BaseModel):
    status: str                       # drafted | passed | failed | no_repository | error
    feature_text: Optional[str] = None
    validation: Optional[dict] = None
    step_results: Optional[list[dict]] = None
    notes: list[str] = []
    root_cause: Optional[str] = None
    repo_field_count: int = 0
