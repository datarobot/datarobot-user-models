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

_The Dockerfile.local should be used when customizing the Dockerfile or building locally._

### Creating models for this environment

To use this environment, your custom model archive will typically contain a `custom.py` file containing the necessary hooks, as well as other files needed for your workflow. You can implement the hook functions such as `load_model` and `score_unstructured`, as documented [here](../../custom_model_runner/README.md)

Within your `custom.py` code, by importing the necessary dependencies found in this environment, you can implement your Python code under the related custom hook functions, to build your GenAI workflows.

If you need additional dependencies, you can add those packages in your `requirements.txt` file that you include within your custom model archive and DataRobot will make them available to your custom Python code after you build the environment.
