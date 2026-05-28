from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from .config import settings

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session_factory() as session:
        yield session


async def init_db(max_retries: int = 15, delay: float = 2.0) -> None:
    """
    Create all tables, with retry logic for the startup race between the API
    container and the PostgreSQL container.

    Docker's depends_on healthcheck fires pg_isready inside the db container,
    but the Docker network's embedded DNS may not have propagated the 'db'
    hostname to the api container in time for the very first connection attempt.
    Retrying here means the service comes up cleanly even after an unclean DB
    shutdown (crash recovery takes a few seconds before pg_isready passes).
    """
    import asyncio
    import logging
    logger = logging.getLogger(__name__)

    from . import models  # noqa: F401 — ensure models are registered before create_all

    for attempt in range(1, max_retries + 1):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema ready (attempt %d)", attempt)
            return
        except Exception as exc:
            if attempt == max_retries:
                raise
            logger.warning(
                "Database not ready (attempt %d/%d): %s — retrying in %.1fs",
                attempt, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
