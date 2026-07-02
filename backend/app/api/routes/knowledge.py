"""Application Knowledge routes.

GET  /knowledge/items?kind=&transaction_number=   — browse what the platform has
                                                    learned (screens, lessons,
                                                    episodes)
GET  /knowledge/search?q=...                      — semantic search over memory
GET  /knowledge/export/finetuning                 — chat-format JSONL of the whole
                                                    knowledge base, ready for a
                                                    future Azure OpenAI fine-tune
"""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.agents.knowledge_agent import export_finetuning_jsonl, retrieve
from app.core.config import get_settings
from app.core.database import get_db
from app.models.knowledge import KnowledgeItem

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/items")
def list_items(kind: str | None = None, transaction_number: str | None = None,
               limit: int = 100, db: Session = Depends(get_db)):
    query = db.query(KnowledgeItem).order_by(KnowledgeItem.created_at.desc())
    if kind:
        query = query.filter(KnowledgeItem.kind == kind)
    if transaction_number:
        query = query.filter(KnowledgeItem.transaction_number == transaction_number)
    return [
        {
            "id": i.id, "kind": i.kind, "transaction_number": i.transaction_number,
            "screen_name": i.screen_name, "title": i.title, "content": i.content,
            "source": i.source, "created_at": i.created_at,
        }
        for i in query.limit(min(limit, 500)).all()
    ]


@router.get("/search")
def search(q: str, n: int = 5):
    return {"query": q, "results": retrieve(q, n=n)}


@router.get("/export/finetuning")
def export_finetuning(db: Session = Depends(get_db)):
    settings = get_settings()
    path = Path(settings.export_dir) / "knowledge_finetuning.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = export_finetuning_jsonl(db, path)
    return FileResponse(path, media_type="application/jsonl",
                        filename=f"nbc_knowledge_finetuning_{count}_records.jsonl")
