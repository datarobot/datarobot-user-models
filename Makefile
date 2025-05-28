
PYTEST_IGNORES := -W ignore::pytest.PytestCollectionWarning

########################################
##@ General
default: help

help: ## Print this message
	@echo "====================="
	@echo "🛠  Available Commands"
	@echo "====================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

prereq: ## Install the development prerequisites
	pip install -r ./custom_model_runner/requirements.txt -r requirements_test_unit.txt -r requirements_test.txt -r requirements_lint.txt
	pip install -e custom_model_runner/

clean: cov-clean ## Remove build artifacts
	\rm -rf build
	\rm -rf dist
	\rm -rf lib
	\rm -f results-py3.xml
	find . -type d -name '*.egg-info' | xargs rm -rf
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name '*.pyc' -delete

update-env: ## Update execution-environment version
	./tools/env_version_update.py

########################################
##@ Lint
lint: ## Run linting
	black --check --diff .

delint: ## Attempt to fix lint issues
	black .

# NOTE: pylint currently yields lots of errors, so it is in a separate target
pylint: ## Run pylint
	pylint custom_model_runner/

########################################
##@ Test
tests: unit-test integration-tests functional-tests ## Run all tests

# cleanup coverage artifacts
cov-clean:
	rm -rf .coverage htmlcov/

cov: coverage
coverage: cov-clean ## Measure unit-test code coverage
	pytest -v $(PYTEST_IGNORES) --cov-report term --cov-report html --cov custom_model_runner/ tests/unit
	@echo "open file://`pwd`/htmlcov/index.html"

unit-test: ## Run all unit tests
	pytest -v $(PYTEST_IGNORES) tests/unit

integration-tests: ## Run all integration tests
	pytest -v $(PYTEST_IGNORES) tests/integration

functional-tests: ## Run all the functional tests
	pytest -v $(PYTEST_IGNORES) tests/functional
