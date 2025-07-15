"""Security headers middleware."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware stub."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response."""
        # TODO: Implement security headers logic
        response = await call_next(request)
        return response
