"""Rate limiting middleware for RAG API"""
import time
from collections import defaultdict
from typing import DefaultDict, List

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting.

    For production, use Redis-backed rate limiting.
    """

    def __init__(
        self,
        app,
        calls: int = 100,
        period: int = 60,
    ):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application
            calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests: DefaultDict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Clean old requests outside the time window
        now = time.time()
        self.requests[client_ip] = [
            timestamp
            for timestamp in self.requests[client_ip]
            if now - timestamp < self.period
        ]

        # Check rate limit
        if len(self.requests[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": f"{self.calls} requests per {self.period} seconds",
                    "retry_after": int(self.period - (now - self.requests[client_ip][0])),
                },
            )

        # Record this request
        self.requests[client_ip].append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self.calls - len(self.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(now + self.period))

        return response
