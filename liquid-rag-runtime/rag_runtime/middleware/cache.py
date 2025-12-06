"""Response caching middleware for RAG API"""
import hashlib
import json
import time
from typing import Dict, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class CacheMiddleware(BaseHTTPMiddleware):
    """
    In-memory response caching for identical requests.

    For production, use Redis or Memcached for distributed caching.
    """

    def __init__(
        self,
        app,
        ttl: int = 300,  # 5 minutes default
        max_size: int = 1000,  # Maximum cached items
    ):
        """
        Initialize cache middleware.

        Args:
            app: FastAPI application
            ttl: Time-to-live in seconds (default 5 minutes)
            max_size: Maximum number of cached items
        """
        super().__init__(app)
        self.ttl = ttl
        self.max_size = max_size
        # Cache structure: key -> (response_body, headers, status_code, timestamp)
        self.cache: Dict[str, Tuple[bytes, dict, int, float]] = {}
        self.access_times: Dict[str, float] = {}  # For LRU eviction

    def _should_cache(self, request: Request) -> bool:
        """Determine if request should be cached."""
        # Only cache POST requests to /ask and /search endpoints
        if request.method != "POST":
            return False

        path = request.url.path
        cacheable_paths = ["/ask", "/ask/simple", "/search"]

        return any(path.endswith(p) for p in cacheable_paths)

    def _generate_cache_key(self, request: Request, body: bytes) -> str:
        """Generate cache key from request path and body."""
        # Combine path and body for unique key
        key_data = f"{request.url.path}:{body.decode('utf-8')}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _evict_if_needed(self):
        """Evict oldest items if cache is full (LRU)."""
        if len(self.cache) >= self.max_size:
            # Find and remove least recently used item
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached item is still valid."""
        return (time.time() - timestamp) < self.ttl

    async def dispatch(self, request: Request, call_next):
        # Check if request should be cached
        if not self._should_cache(request):
            return await call_next(request)

        # Read request body
        body = await request.body()

        # Generate cache key
        cache_key = self._generate_cache_key(request, body)

        # Check cache
        if cache_key in self.cache:
            cached_body, cached_headers, status_code, timestamp = self.cache[cache_key]

            # Check if cache is still valid
            if self._is_cache_valid(timestamp):
                # Update access time for LRU
                self.access_times[cache_key] = time.time()

                # Return cached response
                response = Response(
                    content=cached_body,
                    status_code=status_code,
                    headers=cached_headers,
                )
                response.headers["X-Cache"] = "HIT"
                response.headers["X-Cache-Age"] = str(int(time.time() - timestamp))
                return response
            else:
                # Cache expired, remove it
                del self.cache[cache_key]
                del self.access_times[cache_key]

        # Cache miss - process request
        # Re-create request with body since we consumed it
        async def receive():
            return {"type": "http.request", "body": body}

        request._receive = receive

        response = await call_next(request)

        # Only cache successful responses (2xx status codes)
        if 200 <= response.status_code < 300:
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            # Store in cache
            self._evict_if_needed()

            # Extract headers (exclude some that shouldn't be cached)
            headers = dict(response.headers)
            headers_to_remove = ["set-cookie", "date", "server"]
            for header in headers_to_remove:
                headers.pop(header, None)

            self.cache[cache_key] = (
                response_body,
                headers,
                response.status_code,
                time.time(),
            )
            self.access_times[cache_key] = time.time()

            # Return response with cache miss header
            new_response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=headers,
            )
            new_response.headers["X-Cache"] = "MISS"
            return new_response

        # Don't cache error responses
        return response

    def clear_cache(self):
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        now = time.time()
        valid_items = sum(
            1 for timestamp in [v[3] for v in self.cache.values()]
            if self._is_cache_valid(timestamp)
        )

        return {
            "total_items": len(self.cache),
            "valid_items": valid_items,
            "expired_items": len(self.cache) - valid_items,
            "max_size": self.max_size,
            "utilization": (len(self.cache) / self.max_size) * 100,
            "ttl_seconds": self.ttl,
        }
