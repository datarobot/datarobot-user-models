# Python 3 GenAI Agents Drop-In Template Environment

This template environment can be used to create GenAI-powered custom models and includes common dependencies for
workflows using CrewAI, LangGraph, Llama-Index and other agentic workflows.

Additionally, this environment is fully compatible with `Codespaces` and `Notebooks` in the DataRobot platform.

## Supported Libraries

This environment is built for python 3 and has support for the following libraries.
For specific version information and the complete list of included packages, see [requirements](requirements.txt).

- crewai
- langgraph
- langchain
- llama-index
- openai
- numpy
- pandas

## Instructions

1. From the terminal, run `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/python311_genai_agents/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created
in step 1.

*The Dockerfile.local should be used when customizing the Dockerfile or building locally.*

## [Development] Updating dependencies in Agentic Execution Env

> ⚠️ **WARNING:** `recipe-datarobot-agent-templates` repo must be considered source of truth for all dependencies. Whenever you need to update deps, please merge your changes there first, release new version of agentic template, and follow steps below to synchronize changes afterwards. Please do not update deps directly here. 

### Updating agentic template component dependencies.
1. Within `recipe-datarobot-agent-templates` perform necessary updates to component dependencies, either manually or full lock upgrade if needed, for example.
```bash
  cd <recipe-datarobot-agent-templates>
  # Targeted update.
  cd agent_crewai
  uv lock --upgrade-package <package-name>

  # Bump all dependencies to latest.
  task update
```
2. Export context from `recipe-datarobot-agent-templates`. Replace `path/to/` with the approprite path of your local environment:
```bash
  cd <recipe-datarobot-agent-templates>
  task execenv:update-context AGENT_PATH=/path/to/datarobot-user-models/public_dropin_environments/python311_genai_agents
```

This will synchronize `app_components` folder with `toml/lock` files for all components.

### Updating codespace requirements.
Codespace requirements live in `requirements.in` and are actually installed into venv in image. They should also be kept in sync with `recipe-datarobot-agent-templates`, and only updated/modified if absolutely necessary.
1. Update `requirements.in` in `recipe-datarobot-agent-templates`.
```bash
  cd <recipe-datarobot-agent-templates>/docker_context
  vim requirements.in
```
2. Compile dependencies into requirements.txt (Python 3.11 and `pip-tools` are required for that):
```bash
  pip-compile --no-annotate --no-emit-index-url --no-emit-trusted-host --output-file=requirements.txt requirements.in
```
3. Copy both files to your local environment manually:
```bash
  cp -vf <recipe-datarobot-agent-templates>/docker_context/requirements* </path/to>
```
