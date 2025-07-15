"""Rate limiting middleware."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware stub."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response."""
        # TODO: Implement rate limiting logic
        response = await call_next(request)
        return response
