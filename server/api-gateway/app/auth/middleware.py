"""Authentication middleware."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware stub."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response."""
        # TODO: Implement authentication logic
        response = await call_next(request)
        return response
