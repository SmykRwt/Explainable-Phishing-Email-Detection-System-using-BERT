import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.database.session import init_db
from backend.app.api.endpoints.analysis import router as analysis_router
from backend.app.api.endpoints.history import router as history_router

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("phishing_platform")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing Phishing Intelligence Platform backend...")
    
    # Create tables
    await init_db()
    
    yield
    
    # Shutdown actions
    logger.info("Shutting down Phishing Intelligence Platform backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Explainable Phishing Detection Platform using ML, BERT, Rules, and GenAI summaries.",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(analysis_router, prefix=f"{settings.API_V1_STR}/analyze", tags=["analysis"])
app.include_router(history_router, prefix=settings.API_V1_STR, tags=["dashboard"])

@app.get("/", tags=["health"])
def root_route():
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME}",
        "docs_url": "/docs",
        "status": "active"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
