# Python 3 GenAI Agents Drop-In Template Environment

This template environment can be used to create GenAI-powered custom models and includes common dependencies for
workflows using CrewAI, LangGraph, Llama-Index and other agentic workflows.

Additionally, this environment is fully compatible with `Codespaces` and `Notebooks` in the DataRobot platform.

## Supported Libraries
For specific version information and the complete list of included packages, see [pyproject.toml](pyproject.toml).

## Instructions

1. From the terminal, run `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/python311_genai_agents/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created in step 1.

_The Dockerfile.local should be used when customizing the Dockerfile or building locally._

## [Development] Synchronizing `pyproject.toml` and other files with `af-component-agents` [Preferred method]
From within the `af-component-agents` repo run the following while replacing `path/to/` with the approprite path of your local environment:
```bash
task docker_update_reqs AGENT_PATH=/path/to/datarobot-user-models/public_dropin_environments/python311_genai_agents
```

This will:
- Synchronize the `pyproject.toml` to the latest unified requirements
- Upgrade the `uv.lock` file
- Update the `requirements.txt` file so it properly displays in the Execution Environment UI.
