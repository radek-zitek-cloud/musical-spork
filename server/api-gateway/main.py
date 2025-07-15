"""
Musical Spork API Gateway - Main Application Entry Point

This is the FastAPI application entry point for the Universal API Gateway.
It handles application lifecycle, middleware setup, and route registration.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from app.core.config import settings
from app.core.redis import redis_manager
from app.core.exceptions import setup_exception_handlers
from app.api.v1 import auth, users, health, admin
from app.auth.middleware import AuthMiddleware
from app.rate_limiting.middleware import RateLimitMiddleware
from app.caching.middleware import CacheMiddleware
from app.monitoring.middleware import MonitoringMiddleware
from app.security.headers import SecurityHeadersMiddleware
from app.monitoring.metrics import setup_metrics
from app.monitoring.logging import setup_logging
from app.monitoring.tracing import setup_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    setup_logging()
    setup_metrics()
    setup_tracing()
    
    await redis_manager.connect()
    
    # TODO: Initialize service registry
    # TODO: Start health check scheduler
    
    yield
    
    # Shutdown
    await redis_manager.disconnect()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Musical Spork API Gateway",
        description="Universal API Gateway for Musical Spork microservices ecosystem",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (order matters!)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MonitoringMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(CacheMiddleware)
    app.add_middleware(AuthMiddleware)

    # Exception handlers
    setup_exception_handlers(app)

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower(),
    )
