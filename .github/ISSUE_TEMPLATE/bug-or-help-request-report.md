---
name: Bug or Help Request Report
about: This template is intended for bug or help request reports
title: "[REPORT]"
labels: bug, help wanted
assignees: AstraBert
---

### Checklist

Before reporting bugs or help requests, please fill out the following checklist:

- [ ] I set the `.env` file with the necessary keys, including OPENAI_API_KEY, LLAMACLOUD_API_KEY, ELEVENLABS_API_KEY
- [ ] My API keys are all functioning and have at least some credit balance
- [ ] I ran both the scripts in the `tools` directory
- [ ] I checked that my LlamaCloud Extract Agent and my LlamaCloud Index has been correctly created and are functioning (using the playground functionality in the [UI](https://cloud.llamaindex.ai)
- [ ] I activated the virtual environment and ensured that all the dependencies are installed properly
- [ ] As a first step, I launched the docker services through `docker compose up -d`
- [ ] As a second step, I launched the MCP server through `uv run src/notebookllama/server.py`
- [ ] As a third and last step, I launched the Streamlit app through `strealit run src/notebookllama/Home.py`

### Issue Description

My issue is...

### Relevant Traceback

```text

```

### Other details

**OS**:
**uv version**:
**streamlit version**:
**fastmcp version**:
**llama-index-workflows version**:
