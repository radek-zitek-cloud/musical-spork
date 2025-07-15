"""Monitoring middleware."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Monitoring middleware stub."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response."""
        # TODO: Implement monitoring logic
        response = await call_next(request)
        return response
