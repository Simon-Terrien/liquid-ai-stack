"""Request logging middleware for RAG API"""
import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("rag_api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing and status"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"← {response.status_code} {request.url.path} "
            f"({duration:.3f}s)"
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"

        return response
