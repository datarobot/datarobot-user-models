# Security Audit Checklist: FastAPI Migration

This document provides a comprehensive security audit checklist for the Flask → FastAPI migration.

## Pre-Migration Security Audit

### 1. Dependency Security

| Check | Status | Notes |
|-------|--------|-------|
| [ ] Run `pip-audit` on new dependencies | | `fastapi`, `uvicorn`, `httpx` |
| [ ] Check CVE database for known vulnerabilities | | |
| [ ] Pin exact versions in requirements.txt | | Avoid `>=` for security-critical deps |
| [ ] Verify transitive dependencies | | Use `pip-compile` or `poetry lock` |

```bash
# Run security audit
pip install pip-audit
pip-audit -r requirements.txt
```

### 2. Input Validation

| Endpoint | Flask Behavior | FastAPI Behavior | Parity Verified |
|----------|---------------|------------------|-----------------|
| `/predict/` | Manual validation | Pydantic models | [ ] |
| `/transform/` | Manual validation | Pydantic models | [ ] |
| `/predictUnstructured/` | Manual validation | Pydantic models | [ ] |
| File uploads | Werkzeug limits | Starlette limits | [ ] |

**Required Tests:**
```python
# Test for oversized payloads
def test_max_upload_size_enforced():
    large_file = b"x" * (MAX_UPLOAD_SIZE + 1)
    response = client.post("/predict/", files={"X": large_file})
    assert response.status_code == 413

# Test for malformed content-type
def test_malformed_content_type():
    response = client.post("/predict/", 
        headers={"Content-Type": "application/json; charset=utf-8; extra=malicious"},
        content=b"{}")
    assert response.status_code in [200, 400]  # Not 500
```

### 3. Rate Limiting & DoS Protection

#### Slow Loris Attack Mitigation

**Uvicorn Configuration:**
```python
# In config.py
@dataclass
class UvicornConfig:
    # ... existing fields ...
    
    # Slow loris protection
    timeout_notify: int = 30  # Time to wait for headers
    limit_concurrency: int = 100  # Max concurrent connections
    
    @classmethod
    def from_runtime_params(cls) -> "UvicornConfig":
        # ... existing code ...
        
        # Slow loris protection
        if RuntimeParameters.has("DRUM_UVICORN_HEADER_TIMEOUT"):
            config.timeout_notify = int(RuntimeParameters.get("DRUM_UVICORN_HEADER_TIMEOUT"))
```

**Nginx/Proxy Configuration (recommended):**
```nginx
# /etc/nginx/conf.d/drum.conf
client_body_timeout 60s;
client_header_timeout 60s;
send_timeout 60s;
keepalive_timeout 65s;
limit_req_zone $binary_remote_addr zone=drum_limit:10m rate=100r/s;

location / {
    limit_req zone=drum_limit burst=200 nodelay;
    proxy_pass http://drum_server;
}
```

### 4. CORS Security

| Configuration | Secure | Insecure |
|---------------|--------|----------|
| `allow_origins=["*"]` | ❌ | ✅ Open to all |
| `allow_origins=["https://app.datarobot.com"]` | ✅ | |
| `allow_credentials=True` with `*` | ❌ CRITICAL | Browser blocks, but misconfiguration risk |

**Secure CORS Configuration:**
```python
# In server.py - create_fastapi_app()
def create_fastapi_app() -> FastAPI:
    # ...
    if RuntimeParameters.has("DRUM_CORS_ENABLED"):
        if str(RuntimeParameters.get("DRUM_CORS_ENABLED")).lower() in ("true", "1", "yes"):
            origins = ["*"]
            allow_credentials = False  # Default to False for security
            
            if RuntimeParameters.has("DRUM_CORS_ORIGINS"):
                origins_str = str(RuntimeParameters.get("DRUM_CORS_ORIGINS"))
                if origins_str != "*":
                    origins = [o.strip() for o in origins_str.split(",")]
                    allow_credentials = True  # Safe with specific origins
            
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=allow_credentials,
                allow_methods=["GET", "POST"],  # Restrict to needed methods
                allow_headers=["X-Auth-Token", "Content-Type", "X-Request-ID"],
                max_age=3600,  # Cache preflight for 1 hour
            )
```

### 5. SSL/TLS Configuration

#### Recommended Cipher Suites (2024+)

```python
# In config.py
SECURE_CIPHERS = (
    "ECDHE+AESGCM:"
    "DHE+AESGCM:"
    "ECDHE+CHACHA20:"
    "DHE+CHACHA20:"
    "!aNULL:!MD5:!DSS:!RC4:!3DES"
)

@dataclass
class UvicornConfig:
    # ...
    ssl_ciphers: str = SECURE_CIPHERS
    ssl_version: int = ssl.PROTOCOL_TLS_SERVER  # TLS 1.2+
```

**Testing SSL Configuration:**
```bash
# Test with OpenSSL
openssl s_client -connect localhost:8080 -tls1_2
openssl s_client -connect localhost:8080 -tls1_3

# Check for weak ciphers
nmap --script ssl-enum-ciphers -p 8080 localhost
```

### 6. Header Security

**Required Security Headers Middleware:**
```python
# In middleware.py
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy for /docs endpoint
        if request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
                "img-src 'self' data: cdn.jsdelivr.net"
            )
        
        return response
```

### 7. Error Handling Security

**Never expose internal errors to clients:**
```python
# In app.py
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log full error internally
    logger.exception("Unhandled exception: %s", exc)
    
    # Return sanitized error to client
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
    )

# Never do this:
# return JSONResponse(content={"error": str(exc), "traceback": traceback.format_exc()})
```

### 8. Authentication Security

**Token Timing Attack Prevention:**
```python
# In custom_fastapi.py example for users
import secrets

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.rstrip("/") in ["/ping", "/health", "/livez", "/readyz"]:
            return await call_next(request)
        
        token = request.headers.get("X-Auth-Token", "")
        expected_token = os.environ.get("AUTH_TOKEN", "")
        
        # Use constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(token, expected_token):
            return JSONResponse(
                content={"message": "Unauthorized"},
                status_code=401
            )
        
        return await call_next(request)
```

---

## Post-Migration Security Verification

### Automated Security Tests

```python
# tests/security/test_security.py
import pytest
from httpx import AsyncClient

class TestSecurityHeaders:
    async def test_no_server_header_leak(self, client: AsyncClient):
        """Ensure server version is not exposed."""
        response = await client.get("/ping")
        assert "uvicorn" not in response.headers.get("server", "").lower()
    
    async def test_x_frame_options(self, client: AsyncClient):
        response = await client.get("/ping")
        assert response.headers.get("X-Frame-Options") == "DENY"
    
    async def test_content_type_options(self, client: AsyncClient):
        response = await client.get("/ping")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

class TestInputValidation:
    async def test_sql_injection_in_params(self, client: AsyncClient):
        """Ensure SQL-like payloads don't cause errors."""
        response = await client.post("/predict/", 
            json={"data": "'; DROP TABLE users; --"})
        assert response.status_code != 500
    
    async def test_path_traversal(self, client: AsyncClient):
        """Ensure path traversal is blocked."""
        response = await client.get("/../../../etc/passwd")
        assert response.status_code == 404

class TestDoSProtection:
    async def test_large_payload_rejected(self, client: AsyncClient):
        """Ensure oversized payloads are rejected."""
        large_payload = b"x" * (100 * 1024 * 1024 + 1)  # 100MB + 1
        response = await client.post("/predict/", content=large_payload)
        assert response.status_code == 413
    
    async def test_slow_request_timeout(self, client: AsyncClient):
        """Ensure slow requests are timed out."""
        # This requires a special test setup with delayed responses
        pass
```

---

## Security Sign-Off Checklist

Before each milestone:

- [ ] All automated security tests pass
- [ ] `pip-audit` shows no HIGH/CRITICAL vulnerabilities
- [ ] CORS configuration reviewed for specific origins
- [ ] SSL/TLS configuration tested with modern ciphers
- [ ] Error responses do not leak stack traces
- [ ] Rate limiting configured at proxy level
- [ ] Security headers present on all responses
- [ ] Authentication uses constant-time comparison
- [ ] File upload limits enforced
- [ ] Request timeout configured

**Sign-off:**
- Security Engineer: _________________ Date: _______
- Tech Lead: _________________ Date: _______
