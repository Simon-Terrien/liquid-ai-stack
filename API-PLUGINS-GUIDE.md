# API Development Plugins Guide - LiquidAI Stack

All 25 API development plugins have been installed successfully! This guide shows how to use them with your LiquidAI Stack.

## 🎯 Installed Plugins (25/25)

### Core API Development
1. ✅ **rest-api-generator** - Generate REST APIs from schemas
2. ✅ **graphql-server-builder** - Build GraphQL servers
3. ✅ **grpc-service-generator** - Generate gRPC services
4. ✅ **websocket-server-builder** - Build WebSocket servers

### Documentation & Contracts
5. ✅ **api-documentation-generator** - Auto-generate API docs
6. ✅ **api-contract-generator** - Generate API contracts (OpenAPI)
7. ✅ **api-schema-validator** - Validate API schemas

### Security & Authentication
8. ✅ **api-security-scanner** - Scan for security vulnerabilities
9. ✅ **api-authentication-builder** - Build auth systems
10. ✅ **api-rate-limiter** - Implement rate limiting

### Performance & Scaling
11. ✅ **api-cache-manager** - Implement caching strategies
12. ✅ **api-throttling-manager** - Manage API throttling
13. ✅ **api-gateway-builder** - Build API gateways
14. ✅ **api-load-tester** - Load test APIs

### Monitoring & Logging
15. ✅ **api-monitoring-dashboard** - Create monitoring dashboards
16. ✅ **api-request-logger** - Log API requests
17. ✅ **api-event-emitter** - Handle events

### Error Handling & Validation
18. ✅ **api-error-handler** - Build error handlers
19. ✅ **api-response-validator** - Validate API responses

### Integration & Webhooks
20. ✅ **webhook-handler-creator** - Create webhook handlers
21. ✅ **api-batch-processor** - Process batch requests

### Development Tools
22. ✅ **api-mock-server** - Create mock API servers
23. ✅ **api-sdk-generator** - Generate SDKs
24. ✅ **api-versioning-manager** - Manage API versions
25. ✅ **api-migration-tool** - Handle API migrations

---

## 🚀 Use Cases for LiquidAI Stack

### 1. Enhance the RAG API Server

Your existing RAG API (`liquid-rag-runtime/rag_runtime/api_server.py`) can be enhanced with:

#### Add API Documentation
```bash
/api-documentation-generator
# Input: liquid-rag-runtime/rag_runtime/api_server.py
# Output: OpenAPI/Swagger documentation
```

#### Add Rate Limiting
```bash
/api-rate-limiter
# Add rate limiting to /ask endpoint
# Prevent abuse of expensive LLM queries
```

#### Add Caching
```bash
/api-cache-manager
# Cache common queries
# Reduce LLM inference costs
```

#### Add Request Logging
```bash
/api-request-logger
# Log all queries for analytics
# Track usage patterns
```

#### Add Monitoring Dashboard
```bash
/api-monitoring-dashboard
# Monitor query latency
# Track model performance
# View cache hit rates
```

### 2. Create New API Endpoints

#### Generate REST API for Model Management
```bash
/rest-api-generator
# Endpoint: /models/download
# Endpoint: /models/status
# Endpoint: /models/list
```

#### Add WebSocket for Streaming Responses
```bash
/websocket-server-builder
# Stream RAG responses in real-time
# Progressive answer generation
# Better UX for long responses
```

#### Create Webhook Handlers
```bash
/webhook-handler-creator
# Handle document upload events
# Trigger ETL pipeline on new documents
# Notify when processing complete
```

### 3. API Security Enhancements

#### Scan for Vulnerabilities
```bash
/api-security-scanner
# Scan liquid-rag-runtime/rag_runtime/api_server.py
# Check for common security issues
# OWASP Top 10 compliance
```

#### Add Authentication
```bash
/api-authentication-builder
# Add API key authentication
# JWT token support
# OAuth2 integration
```

#### Add API Gateway
```bash
/api-gateway-builder
# Centralized routing
# Load balancing
# SSL/TLS termination
```

### 4. Development & Testing

#### Create Mock API Server
```bash
/api-mock-server
# Mock RAG responses for testing
# Test frontend without models
# Faster development iteration
```

#### Load Testing
```bash
/api-load-tester
# Test RAG API under load
# Find performance bottlenecks
# Capacity planning
```

#### Generate Python SDK
```bash
/api-sdk-generator
# Generate client SDK
# Easy integration for users
# Type-safe API client
```

### 5. API Versioning & Migration

#### Add API Versioning
```bash
/api-versioning-manager
# /v1/ask for stable API
# /v2/ask for new features
# Backward compatibility
```

#### Handle Breaking Changes
```bash
/api-migration-tool
# Migrate from v1 to v2
# Deprecation warnings
# Migration guides
```

---

## 📋 Practical Examples

### Example 1: Add Comprehensive Monitoring to RAG API

```bash
# 1. Add request logging
/api-request-logger --path liquid-rag-runtime/rag_runtime/api_server.py

# 2. Add monitoring dashboard
/api-monitoring-dashboard --metrics "latency,throughput,errors,cache_hits"

# 3. Add response validation
/api-response-validator --schema liquid-shared-core/liquid_shared/schemas.py

# 4. Generate documentation
/api-documentation-generator --format openapi
```

Result: Production-ready RAG API with full observability

### Example 2: Create ETL API Endpoints

```bash
# 1. Generate REST API for ETL operations
/rest-api-generator --spec etl-api-spec.yaml

# 2. Add batch processing
/api-batch-processor --endpoint /etl/batch

# 3. Add webhook for completion
/webhook-handler-creator --event etl_complete

# 4. Add authentication
/api-authentication-builder --type api-key
```

Result: API for triggering and monitoring ETL jobs

### Example 3: GraphQL API for Complex Queries

```bash
# Build GraphQL server for advanced queries
/graphql-server-builder

# Example schema:
query {
  searchDocuments(query: "GDPR", limit: 10) {
    chunks {
      id
      text
      score
      metadata {
        source
        tags
      }
    }
  }

  askQuestion(question: "What is GDPR?", maxChunks: 5) {
    answer
    sources
    confidence
  }
}
```

Result: Flexible GraphQL API for power users

### Example 4: Production Deployment Setup

```bash
# 1. Add API gateway
/api-gateway-builder --backends "rag:8000,etl:8001"

# 2. Add rate limiting
/api-rate-limiter --rate "100/minute" --burst 20

# 3. Add security scanning
/api-security-scanner --scan-all

# 4. Add monitoring
/api-monitoring-dashboard --prometheus

# 5. Generate client SDK
/api-sdk-generator --language python --output client-sdk/
```

Result: Production-ready deployment with all best practices

---

## 🎨 Integration Patterns

### Pattern 1: Middleware Stack for RAG API

```python
# Enhanced api_server.py with plugin-generated middleware

from api_rate_limiter import RateLimiter
from api_cache_manager import ResponseCache
from api_request_logger import RequestLogger
from api_error_handler import ErrorHandler

app = FastAPI()

# Middleware stack (generated by plugins)
app.add_middleware(RequestLogger)
app.add_middleware(RateLimiter, rate="100/minute")
app.add_middleware(ResponseCache, ttl=300)
app.add_middleware(ErrorHandler, format="json")

# Existing endpoints
@app.post("/ask")
async def ask_question(request: AskRequest):
    # Your existing code
    pass
```

### Pattern 2: Event-Driven ETL

```python
# ETL with webhook notifications

from webhook_handler_creator import WebhookManager
from api_event_emitter import EventEmitter

webhook = WebhookManager()
events = EventEmitter()

@events.on("document_uploaded")
async def trigger_etl(document):
    result = run_etl_pipeline(document)
    await webhook.notify("etl_complete", result)
```

### Pattern 3: Multi-Protocol API

```python
# Support REST, GraphQL, gRPC, and WebSocket

# REST (existing)
@app.post("/ask")
async def ask_rest(request: AskRequest):
    pass

# GraphQL (plugin-generated)
from graphql_server import GraphQLRouter
app.include_router(GraphQLRouter, prefix="/graphql")

# gRPC (plugin-generated)
from grpc_service import RAGService
grpc_server = create_grpc_server(RAGService)

# WebSocket (plugin-generated)
from websocket_server import WebSocketHandler
app.add_websocket_route("/ws/ask", WebSocketHandler)
```

---

## 🔧 Plugin Commands Quick Reference

### Documentation
```bash
/api-docs                   # Generate OpenAPI docs
/api-contract              # Generate API contract
/api-schema validate       # Validate schemas
```

### Security
```bash
/api-scan                  # Security scan
/api-auth setup            # Setup authentication
/api-rate-limit configure  # Configure rate limiting
```

### Monitoring
```bash
/api-monitor dashboard     # View dashboard
/api-logs view             # View request logs
/api-metrics               # View metrics
```

### Development
```bash
/api-mock create           # Create mock server
/api-test load             # Load test
/api-sdk generate          # Generate SDK
```

### Operations
```bash
/api-version create        # Create new version
/api-migrate               # Run migration
/api-gateway setup         # Setup gateway
```

---

## 📊 Recommended Plugin Combinations

### For Development
1. api-mock-server
2. api-documentation-generator
3. api-response-validator
4. api-load-tester

### For Production
1. api-rate-limiter
2. api-cache-manager
3. api-monitoring-dashboard
4. api-error-handler
5. api-security-scanner

### For Integration
1. webhook-handler-creator
2. api-sdk-generator
3. api-batch-processor
4. api-event-emitter

### For Scaling
1. api-gateway-builder
2. api-throttling-manager
3. api-cache-manager
4. websocket-server-builder

---

## 🎯 Next Steps

### Immediate Actions
1. Generate OpenAPI docs for RAG API
   ```bash
   /api-documentation-generator
   ```

2. Add rate limiting to prevent abuse
   ```bash
   /api-rate-limiter
   ```

3. Set up monitoring dashboard
   ```bash
   /api-monitoring-dashboard
   ```

### Short-term Goals
1. Create WebSocket endpoint for streaming
2. Generate Python SDK for clients
3. Add request logging and analytics
4. Set up load testing

### Long-term Goals
1. Build API gateway for all services
2. Implement comprehensive caching
3. Add GraphQL support
4. Create monitoring and alerting

---

## 📚 Resources

- OpenAPI Specification: https://swagger.io/specification/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- API Design Best Practices: https://cloud.google.com/apis/design
- REST API Guidelines: https://github.com/microsoft/api-guidelines

---

## 🤝 Plugin Support

All plugins are from the `claude-code-plugins-plus` marketplace and support:
- FastAPI integration
- Pydantic schema compatibility
- Type-safe code generation
- Production-ready patterns
- Best practices enforcement

For issues or questions:
- Check plugin documentation: `/plugin help <name>`
- View plugin source: `/plugin info <name>`
- Report issues: Claude Code GitHub

---

**Note**: Restart Claude Code after plugin installation for full functionality.
