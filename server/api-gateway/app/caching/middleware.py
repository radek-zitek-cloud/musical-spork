"""Cache middleware."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class CacheMiddleware(BaseHTTPMiddleware):
    """Cache middleware stub."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response."""
        # TODO: Implement cache logic
        response = await call_next(request)
        return response
