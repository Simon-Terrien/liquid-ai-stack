"""API metrics tracking for RAG runtime"""
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class APIMetrics:
    """Track API performance metrics"""

    total_requests: int = 0
    total_errors: int = 0
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    response_times: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_request(
        self, duration: float, error: bool = False, cache_hit: bool = False
    ):
        """Record a request with its duration and status"""
        self.total_requests += 1

        if error:
            self.total_errors += 1

        if cache_hit:
            self.total_cache_hits += 1
        else:
            self.total_cache_misses += 1

        self.response_times.append(duration)

        # Keep only last 1000 response times to prevent unbounded growth
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

    def get_stats(self) -> Dict:
        """Get current metrics statistics"""
        if not self.response_times:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p50_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
        else:
            sorted_times = sorted(self.response_times)
            avg_response_time = sum(sorted_times) / len(sorted_times)
            min_response_time = sorted_times[0]
            max_response_time = sorted_times[-1]

            # Calculate percentiles
            def percentile(data, p):
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = int(k) + 1
                if c >= len(data):
                    return data[-1]
                d0 = data[f] * (c - k)
                d1 = data[c] * (k - f)
                return d0 + d1

            p50_response_time = percentile(sorted_times, 50)
            p95_response_time = percentile(sorted_times, 95)
            p99_response_time = percentile(sorted_times, 99)

        error_rate = (
            self.total_errors / self.total_requests if self.total_requests > 0 else 0
        )
        cache_hit_rate = (
            self.total_cache_hits / self.total_requests
            if self.total_requests > 0
            else 0
        )

        uptime = time.time() - self.start_time

        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(error_rate * 100, 2),
            "cache_hit_rate": round(cache_hit_rate * 100, 2),
            "response_times_ms": {
                "avg": round(avg_response_time * 1000, 2),
                "min": round(min_response_time * 1000, 2),
                "max": round(max_response_time * 1000, 2),
                "p50": round(p50_response_time * 1000, 2),
                "p95": round(p95_response_time * 1000, 2),
                "p99": round(p99_response_time * 1000, 2),
            },
            "requests_per_second": round(
                self.total_requests / uptime if uptime > 0 else 0, 2
            ),
        }

    def reset(self):
        """Reset all metrics"""
        self.total_requests = 0
        self.total_errors = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.response_times = []
        self.start_time = time.time()


# Global metrics instance
metrics = APIMetrics()
