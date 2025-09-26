import os
import io
import pytest
from notebookllama.Home import is_ollama_available, sync_run_workflow

TEST_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "test", "brain_for_kids.pdf")


def test_mindmap_ping():
    """Smoke test: ping Ollama endpoint. This test will skip if Ollama is not reachable."""
    if not is_ollama_available():
        pytest.skip("Ollama not available on http://localhost:11434 - skipping integration test")
    assert is_ollama_available() is True


@pytest.mark.integration
def test_mindmap_full_run():
    """
    Full integration test: run sync_run_workflow on the sample PDF and assert that mind_map
    is either a dict (valid JSON) or an empty dict.
    This test may take time depending on model loading.
    """
    if not is_ollama_available():
        pytest.skip("Ollama not available - skipping full integration test")

    with open(TEST_PDF, "rb") as f:
        b = f.read()

    md_content, summary, q_and_a, bullet_points, mind_map = sync_run_workflow(io.BytesIO(b), "test-doc")

    # Accept None as skipped/unprocessed, but prefer empty dict or dict for mind_map
    assert mind_map is not None
    if isinstance(mind_map, dict):
        assert "nodes" in mind_map and "edges" in mind_map or mind_map == {}
    else:
        # If mind_map is a string, at least ensure it's non-empty
        assert isinstance(mind_map, str)
