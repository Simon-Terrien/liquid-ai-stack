#!/usr/bin/env python3
"""
Test script for the enhanced RAG API server.

Demonstrates the production features:
- Rate limiting with headers
- Request logging with timing
- Metrics tracking
- Enhanced health checks
- All RAG endpoints
"""
import requests
import json
import time
from typing import Dict, Any


API_BASE = "http://localhost:8000"


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_response(response: requests.Response):
    """Pretty print response with headers and body."""
    print(f"\nStatus: {response.status_code}")

    # Show rate limit headers
    rate_headers = {
        k: v for k, v in response.headers.items()
        if k.startswith("X-RateLimit") or k == "X-Response-Time"
    }
    if rate_headers:
        print("\nHeaders:")
        for k, v in rate_headers.items():
            print(f"  {k}: {v}")

    # Show body
    try:
        data = response.json()
        print("\nResponse:")
        print(json.dumps(data, indent=2))
    except Exception:
        print(f"\nBody: {response.text}")


def test_health():
    """Test enhanced health endpoint."""
    print_section("1. Enhanced Health Check")

    response = requests.get(f"{API_BASE}/health")
    print_response(response)

    if response.status_code == 200:
        data = response.json()
        print("\n✓ Health check passed")
        print(f"  Uptime: {data.get('uptime_human', 'unknown')}")
        print(f"  GPU: {data.get('gpu', {}).get('available', False)}")
        print(f"  Total requests: {data.get('api_metrics', {}).get('total_requests', 0)}")
    else:
        print("\n✗ Health check failed")


def test_metrics():
    """Test metrics endpoint."""
    print_section("2. Metrics Endpoint")

    response = requests.get(f"{API_BASE}/metrics")
    print_response(response)

    if response.status_code == 200:
        data = response.json()
        print("\n✓ Metrics retrieved")
        print(f"  Total requests: {data.get('total_requests', 0)}")
        print(f"  Error rate: {data.get('error_rate', 0):.2f}%")
        print(f"  Avg response time: {data.get('avg_response_time_ms', 0):.2f}ms")


def test_stats():
    """Test vector store stats."""
    print_section("3. Vector Store Statistics")

    response = requests.get(f"{API_BASE}/stats")
    print_response(response)

    if response.status_code == 200:
        data = response.json()
        print("\n✓ Vector store stats retrieved")
        print(f"  Total chunks: {data.get('total_documents', 0)}")


def test_search():
    """Test search endpoint."""
    print_section("4. Document Search")

    payload = {
        "query": "What is LiquidAI?",
        "top_k": 3,
        "hybrid": True
    }

    print(f"\nQuery: {payload['query']}")
    response = requests.post(f"{API_BASE}/search", json=payload)
    print_response(response)

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Found {data.get('total', 0)} results")
        for i, result in enumerate(data.get('results', [])[:2], 1):
            print(f"\n  Result {i}:")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Source: {result['source']}")
            print(f"    Text: {result['text'][:100]}...")


def test_ask_simple():
    """Test simple RAG endpoint."""
    print_section("5. Simple RAG (Fast)")

    payload = {
        "question": "What are the key features of LiquidAI models?",
        "max_context_chunks": 3,
        "fast_mode": True
    }

    print(f"\nQuestion: {payload['question']}")
    print("Mode: Simple (no agent overhead)")

    start = time.time()
    response = requests.post(f"{API_BASE}/ask/simple", json=payload)
    duration = time.time() - start

    print_response(response)

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Answer generated in {duration:.2f}s")
        print(f"\n  Answer: {data['answer'][:200]}...")
        print(f"  Context chunks used: {data.get('context_used', 0)}")


def test_ask_full():
    """Test full RAG agent endpoint."""
    print_section("6. Full RAG Agent (Sophisticated)")

    payload = {
        "question": "How does the multi-model strategy work?",
        "max_context_chunks": 5,
        "use_hybrid_search": True,
        "fast_mode": True
    }

    print(f"\nQuestion: {payload['question']}")
    print("Mode: Full agent with retrieval tools")

    start = time.time()
    response = requests.post(f"{API_BASE}/ask", json=payload)
    duration = time.time() - start

    print_response(response)

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Answer generated in {duration:.2f}s")
        print(f"\n  Answer: {data['answer'][:200]}...")
        print(f"  Sources: {data.get('sources', [])}")
        print(f"  Confidence: {data.get('confidence', 'N/A')}")
        print(f"  Context used: {data.get('context_used', 0)}")


def test_rate_limiting():
    """Test rate limiting."""
    print_section("7. Rate Limiting Test")

    print("\nSending 5 rapid requests to test rate limiting...")

    for i in range(5):
        response = requests.get(f"{API_BASE}/health")
        remaining = response.headers.get("X-RateLimit-Remaining", "?")
        limit = response.headers.get("X-RateLimit-Limit", "?")

        print(f"  Request {i+1}: {response.status_code} - "
              f"{remaining}/{limit} requests remaining")

        if response.status_code == 429:
            print("\n✓ Rate limit enforced!")
            print(f"  Response: {response.json()}")
            break

        time.sleep(0.1)
    else:
        print("\n✓ All requests succeeded (under rate limit)")


def test_status():
    """Test quick status endpoint."""
    print_section("8. Quick Status Check")

    response = requests.get(f"{API_BASE}/status")
    print_response(response)

    if response.status_code == 200:
        print("\n✓ Status check passed")


def test_api_docs():
    """Check API documentation."""
    print_section("9. API Documentation")

    print(f"\n✓ Interactive docs available at:")
    print(f"  - Swagger UI: {API_BASE}/docs")
    print(f"  - ReDoc: {API_BASE}/redoc")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  Enhanced RAG API Test Suite")
    print("=" * 60)
    print(f"\nTesting API at: {API_BASE}")
    print("\nMake sure the server is running:")
    print("  cd liquid-rag-runtime")
    print("  uv run python -m rag_runtime.api_server_enhanced")

    input("\nPress Enter to start tests...")

    try:
        # Run all tests
        test_health()
        test_metrics()
        test_stats()
        test_search()
        test_ask_simple()
        test_ask_full()
        test_rate_limiting()
        test_status()
        test_api_docs()

        # Final summary
        print_section("Test Summary")
        response = requests.get(f"{API_BASE}/metrics")
        if response.status_code == 200:
            data = response.json()
            print("\nFinal Metrics:")
            print(f"  Total requests: {data.get('total_requests', 0)}")
            print(f"  Total errors: {data.get('total_errors', 0)}")
            print(f"  Error rate: {data.get('error_rate', 0):.2f}%")
            print(f"  Avg response time: {data.get('avg_response_time_ms', 0):.2f}ms")
            print(f"  Min response time: {data.get('min_response_time_ms', 0):.2f}ms")
            print(f"  Max response time: {data.get('max_response_time_ms', 0):.2f}ms")

        print("\n✓ All tests completed!")

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API server")
        print(f"  Make sure the server is running at {API_BASE}")
        print("\n  Start with: uv run python -m rag_runtime.api_server_enhanced")
    except Exception as e:
        print(f"\n✗ Error during tests: {e}")


if __name__ == "__main__":
    main()
