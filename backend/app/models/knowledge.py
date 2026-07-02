"""Application Knowledge model.

One row per thing the platform "learned" about NBC:
  kind="screen"  — what a screen is for, its fields and their business meaning
                   (captured during crawls)
  kind="episode" — what happened in an execution run, in plain language
                   (captured after every execution)
  kind="lesson"  — a failure or self-heal and how it was resolved
                   (captured per step, the highest-value retrieval unit)

Rows are mirrored into the Chroma "app_knowledge" collection so agents can
retrieve relevant memory semantically (the DB row remains the source of truth;
Chroma is a disposable index that can be rebuilt from these rows).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String)  # screen | episode | lesson
    transaction_number: Mapped[str] = mapped_column(String, nullable=True)
    screen_name: Mapped[str] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String, default="rule")  # rule | llm
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
