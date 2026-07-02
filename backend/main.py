import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analytics, crawl, execution, feature_files, generation, knowledge, locator_repository
from app.core.config import get_settings
from app.core.database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(title="NBC Workflow Automation Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


@app.get("/health")
def health():
    return {"status": "ok", "azure_openai_configured": settings.azure_openai_configured}


app.include_router(crawl.router)
app.include_router(feature_files.router)
app.include_router(execution.router)
app.include_router(locator_repository.router)
app.include_router(analytics.router)
app.include_router(generation.router)
app.include_router(knowledge.router)










#add langgraph in requirements.txt
