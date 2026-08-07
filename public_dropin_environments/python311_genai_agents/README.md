# Python 3 GenAI Agents Drop-In Template Environment

This template environment can be used to create GenAI-powered custom models and includes common dependencies for
workflows using CrewAI, LangGraph, Llama-Index and other agentic workflows.

Additionally, this environment is fully compatible with `Codespaces` and `Notebooks` in the DataRobot platform.

## Why this folder is still called `python311_genai_agents`

The environment is now published as **`[DataRobot] Python 3 GenAI Agents`** and builds on Python 3.13. The folder
name is stale on purpose.

Downstream projects fetch this docker context by downloading a zip of this repository and extracting a hard-coded
path, for example:

```bash
unzip -q datarobot-user-models.zip "$TOP_FOLDER/public_dropin_environments/python311_genai_agents/*"
```

Those projects are already cloned into other people's repositories, and we cannot go update every one of them. If
the folder moved, every existing clone that points at `master` would fail to extract anything and the build would
break with a confusing `filename not matched` error.

A symlink does not rescue this. Git archives store a directory symlink as a single entry with no children, so the
`/*` glob above matches nothing and the extraction still fails.

Leaving the folder in place is what keeps those consumers working, and it is also what lets them keep receiving CVE
fixes instead of being pinned to an unmaintained Python 3.11 image. The version-independent name (`Python 3` rather
than `Python 3.13`) means the published environment does not need to be renamed again at the next Python bump. The
folder will be renamed separately once the downstream references have been migrated.

## Supported Libraries
For specific version information and the complete list of included packages, see [pyproject.toml](pyproject.toml).

## Build locally

1. From the terminal, run `tar -czvf py_dropin.tar.gz -C /path/to/public_dropin_environments/python311_genai_agents/ .`
2. Using either the API or from the UI create a new Custom Environment with the tarball created in step 1.

_The Dockerfile.local should be used when customizing the Dockerfile or building locally._

## [Development] Synchronizing `pyproject.toml` and other files with `af-component-agents` [Preferred method]
From within the `af-component-agents` repo run the following while replacing `path/to/` with the approprite path of your local environment:
```bash
task docker_update_reqs AGENT_PATH=../datarobot-user-models/public_dropin_environments/python311_genai_agents
```

This will:
- Synchronize the `pyproject.toml` to the latest unified requirements
- Upgrade the `uv.lock` file
- Update the `requirements.txt` file so it properly displays in the Execution Environment UI.
