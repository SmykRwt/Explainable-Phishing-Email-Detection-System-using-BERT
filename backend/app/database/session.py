import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from backend.app.core.config import settings

logger = logging.getLogger("phishing_platform")

# Retrieve database URL from config or use a local SQLite DB fallback
db_url = settings.DATABASE_URL
if not db_url:
    # Use SQLite relative to the workspace directory
    db_url = "sqlite+aiosqlite:///./phishing_db.sqlite"
    logger.warning("DATABASE_URL is not set. Falling back to local SQLite database.")

engine = create_async_engine(
    db_url,
    # SQLite requires echo=False and check_same_thread=False
    connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    """Dependency to get DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """Initializes the database, creating tables."""
    async with engine.begin() as conn:
        # Import models inside to ensure they register on Base
        from backend.app.database.models import Analysis, RuleTrigger, URLFinding, LLMReport
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")
