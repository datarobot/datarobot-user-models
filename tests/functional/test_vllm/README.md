# vLLM Dockerfile Tests

Validates that the vLLM Docker environment builds correctly and contains all required components.

## Overview

This test suite ensures the vLLM Dockerfile:
- ✅ Builds successfully without errors
- ✅ Contains required components (Python 3.12, vLLM dependencies)
- ✅ Runs with proper security (non-root user)

## Test Structure

```
test_vllm/
├── test_dockerfile.py           # 3 tests
└── README.md                    # This file
```

### Tests

**TestVLLMDockerBuild** - 3 comprehensive tests (requires Docker):

1. `test_image_builds_successfully` - Validates Docker build completes
2. `test_image_has_required_components` - Verifies Python 3.12, virtualenv, paths
3. `test_container_runs_as_non_root` - Security validation (uid 1000)

## Running Tests

### As Part of All Functional Tests (Recommended):
```bash
make functional-tests  # Runs all functional tests including vLLM
```

### Directly with pytest:
```bash
# Run only vLLM tests
pytest tests/functional/test_vllm/test_dockerfile.py -v

# Or as part of all functional tests
pytest tests/functional/ -v
```

## CI/CD Integration

### Jenkins
```groovy
stage('Test vLLM') {
    steps {
        sh 'pytest tests/functional/test_vllm/test_dockerfile.py -v'
    }
}
```

### GitHub Actions
```yaml
- name: Test vLLM Dockerfile
  run: pytest tests/functional/test_vllm/test_dockerfile.py -v
```

### GitLab CI
```yaml
test:vllm:
  script:
    - pytest tests/functional/test_vllm/test_dockerfile.py -v
```

## Performance

| Run Type | Time |
|----------|------|
| First run (with image download) | 5-15 min |
| Cached run | 3-5 min |

The Docker image is built once and reused across all tests in the class.

## Requirements

### On Your Host Machine (to run tests):
- Docker daemon running
- Python 3.8+ (for pytest test runner)
- pytest, docker Python packages

### Inside the Docker Container (what gets tested):
- Python 3.12 (installed by Dockerfile)
- vLLM dependencies (from base image)
- ~15GB free disk space for image

## Troubleshooting

### Docker not running:
```bash
# Error: "Docker is not available or accessible"
# Solution:
# macOS/Windows: Start Docker Desktop
# Linux: sudo systemctl start docker
```

### Insufficient disk space:
```bash
# Error: "no space left on device"
# Solution:
docker system prune -a  # Remove unused images
docker images           # Check current images
df -h                   # Check disk space
```

### Build fails - Base image unavailable:
```bash
# Error: "pull access denied" or "manifest unknown"
# Context: Image pulled from Docker Hub: vllm/vllm-openai:v0.11.0
# Solution:
# 1. Check internet connection
# 2. Verify base image exists: docker pull vllm/vllm-openai:v0.11.0
# 3. If behind corporate proxy, configure Docker proxy settings
# 4. If authentication required, login: docker login
```

### Build fails - Authentication required:
```bash
# Error: "pull access denied for vllm/vllm-openai" with "authentication required"
# Solution:
# 1. Login to Docker Hub: docker login
# 2. Or use credentials: docker login -u <username> -p <password>
# 3. If using private registry, configure credentials in Docker config
```

### Build fails - Dependency installation:
```bash
# Error: "E: Unable to locate package python3.12"
# Context: Dockerfile specifies Python 3.12 installation (line 4)
# Solution:
# 1. Verify base image hasn't changed: docker pull vllm/vllm-openai:v0.11.0
# 2. Check if Dockerfile was modified in public_dropin_gpu_environments/vllm/
# 3. Try rebuilding without cache: docker build --no-cache
```

### Container fails to start:
```bash
# Error: "Container did not start in time"
# Solution:
# 1. Check Docker daemon logs: docker logs <container-id>
# 2. Verify sufficient resources (CPU/memory)
# 3. Check for port conflicts
```


## Design Philosophy

These tests follow best practices:
- **No redundancy**: Each test validates something unique
- **Fast feedback**: Docker build happens once, reused across tests
- **CI-friendly**: Integrates seamlessly with existing test infrastructure
- **Clear failures**: Detailed error messages for debugging


