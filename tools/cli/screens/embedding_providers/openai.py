
"""
Этот файл переименован и теперь реализует только HuggingFaceEmbeddingScreen.
Рекомендуется переименовать файл в huggingface.py для консистентности.
"""
import os
from textual.app import ComposeResult
from textual.widgets import Input

from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_cloud import PipelineCreateEmbeddingConfig_HuggingfaceApiEmbedding

from ..base import ConfigurationScreen


class HuggingFaceEmbeddingScreen(ConfigurationScreen):
    """Configuration screen for HuggingFace embeddings (замена OpenAI)."""

    def get_title(self) -> str:
        return "HuggingFace Embedding Configuration"

    def get_form_elements(self) -> list[ComposeResult]:
        default_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        return [
            Input(
                placeholder="HuggingFace API Token",
                id="api_key",
                password=True,
                classes="form-control",
                value=default_api_key,
            ),
            Input(
                placeholder="Model name",
                id="model",
                classes="form-control",
            ),
        ]

    def process_submission(self) -> None:
        """Handle form submission by creating HuggingFace embedding configuration."""
        api_key = self.query_one("#api_key", Input).value or os.getenv("HUGGINGFACE_API_KEY")
        model = self.query_one("#model", Input).value

        if not api_key:
            self.notify(
                "No API token provided and HUGGINGFACE_API_KEY not set", severity="error"
            )
            return
        if not model:
            self.notify("Model name is required", severity="error")
            return

        try:
            embed_model = HuggingFaceInferenceAPIEmbedding(token=api_key, model_name=model)
            embedding_config = PipelineCreateEmbeddingConfig_HuggingfaceApiEmbedding(
                type="HUGGINGFACE_API_EMBEDDING",
                component=embed_model,
            )
            self.app.config = embedding_config
            self.app.handle_completion(self.app.config)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
