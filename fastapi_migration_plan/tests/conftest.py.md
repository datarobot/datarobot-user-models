# Plan: tests/conftest.py

Support parameterization of server type across the test suite.

## Overview

Update the pytest configuration to allow running the same functional tests against both Flask and FastAPI server implementations. This ensures parity and enables gradual migration.

## Changes:

### 1. Add pytest command-line option

```python
def pytest_addoption(parser):
    """Add custom pytest options."""
    # Existing options...
    
    parser.addoption(
        "--server-type",
        action="store",
        default="flask",
        choices=["flask", "fastapi", "gunicorn", "all"],
        help="Server type to use for functional tests: flask, fastapi, gunicorn, or all",
    )
```

### 2. Add server_type fixture

```python
import pytest
import os


@pytest.fixture(scope="session")
def server_type(request):
    """
    Fixture that returns the selected server type.
    
    Usage in tests:
        def test_something(server_type):
            if server_type == "fastapi":
                # FastAPI-specific test setup
    """
    return request.config.getoption("--server-type")


@pytest.fixture(scope="function")
def set_server_type_env(server_type):
    """
    Fixture that sets the DRUM_SERVER_TYPE environment variable.
    
    This should be used by tests that need the environment variable set
    before starting the DRUM server.
    """
    original_value = os.environ.get("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE")
    
    if server_type != "all":
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = server_type
    
    yield server_type
    
    # Restore original value
    if original_value is not None:
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = original_value
    else:
        os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)
```

### 3. Add parametrized fixture for running tests with all server types

```python
@pytest.fixture(params=["flask", "fastapi"], scope="class")
def server_type_parametrized(request):
    """
    Parametrized fixture that runs tests with both Flask and FastAPI.
    
    Usage:
        class TestSomething:
            @pytest.fixture(scope="class")
            def drum_server(self, server_type_parametrized, ...):
                os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = server_type_parametrized
                # ... start server
    """
    return request.param


def pytest_generate_tests(metafunc):
    """
    Generate test variants based on --server-type option.
    """
    if "server_type_all" in metafunc.fixturenames:
        server_type = metafunc.config.getoption("--server-type")
        if server_type == "all":
            metafunc.parametrize("server_type_all", ["flask", "fastapi", "gunicorn"])
        else:
            metafunc.parametrize("server_type_all", [server_type])
```

### 4. Update DrumServerRun usage pattern

For tests that should run against all server types:

```python
class TestServerFunctionality:
    """Tests that should run against all server types."""
    
@pytest.fixture(params=["", "/my-prefix"])
def url_prefix(request, monkeypatch):
    """
    Fixture that parametrizes tests with and without a URL prefix.
    """
    prefix = request.param
    if prefix:
        monkeypatch.setenv("URL_PREFIX", prefix)
    return prefix


@pytest.fixture(scope="class")
def drum_server(self, resources, custom_model_dir, server_type_parametrized, url_prefix):
    """Start DRUM server with the specified server type and URL prefix."""
    from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars
    from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun
    
    unset_drum_supported_env_vars()
    
    # Set server type
    if server_type_parametrized in ["fastapi", "gunicorn"]:
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = server_type_parametrized
    
    try:
        with DrumServerRun(
            resources.target_types(REGRESSION),
            resources.class_labels(None, REGRESSION),
            custom_model_dir,
        ) as run:
            yield run
    finally:
        os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)

    def test_ping(self, drum_server):
        """Test ping endpoint works on all server types."""
        response = requests.get(drum_server.url_server_address + "/ping/")
        assert response.ok
```

### 5. Skip tests based on server type

```python
import pytest


def skip_if_not_fastapi(server_type):
    """Skip test if not running on FastAPI."""
    return pytest.mark.skipif(
        server_type != "fastapi",
        reason="Test only applicable to FastAPI server"
    )


def skip_if_not_flask(server_type):
    """Skip test if not running on Flask."""
    return pytest.mark.skipif(
        server_type != "flask",
        reason="Test only applicable to Flask server"
    )


# Usage in tests:
class TestFastAPISpecific:
    @pytest.mark.skipif_not_fastapi
    def test_async_endpoint(self, drum_server):
        """Test async-specific functionality."""
        pass
```

## Running Tests

```bash
# Run all tests with Flask (default)
pytest tests/functional/ -v

# Run all tests with FastAPI
pytest tests/functional/ --server-type=fastapi -v

# Run all tests with both Flask and FastAPI
pytest tests/functional/ --server-type=all -v

# Run specific test file with FastAPI
pytest tests/functional/test_drum_server_fastapi.py --server-type=fastapi -v
```

## Tests to Parametrize

The following test files should be updated to use `server_type_parametrized`:

| Test File | Priority | Notes |
|-----------|----------|-------|
| `test_drum_server_failures.py` | High | Error handling should be consistent |
| `test_stats.py` | High | Stats collection is framework-dependent |
| `test_deployment_config.py` | Medium | Config loading should work the same |
| `test_dr_api_access.py` | Medium | API access patterns should match |
| `test_mlops_monitoring.py` | High | Monitoring hooks are framework-specific |
| `test_runtime_parameters.py` | Medium | Params should work across servers |

## Notes:
- The `--server-type=all` option is useful for CI/CD to ensure parity.
- Individual test files can still use framework-specific fixtures.
- The `skip_if_not_*` markers help manage framework-specific tests.
