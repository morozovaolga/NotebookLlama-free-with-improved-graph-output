import os
from textual.app import App

from .config import EmbeddingConfig
from .screens import InitialScreen


class EmbeddingSetupApp(App):
    """Main application for embedding configuration setup."""

    CSS_PATH = "stylesheets/base.tcss"

    def __init__(self):
        super().__init__()
        self.config = EmbeddingConfig(provider="")

    def on_mount(self) -> None:
        self.push_screen(InitialScreen())

    def handle_completion(self, config: EmbeddingConfig) -> None:
        self.exit(config)

    def handle_default_setup(self) -> None:
        # Используем многоязычную модель, поддерживающую русский язык
        multilingual_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        print(f"Выбрана модель для русского и других языков: {multilingual_model}")

        from llama_cloud import PipelineCreateEmbeddingConfig_HuggingfaceApiEmbedding
        from llama_index.embeddings import HuggingFaceEmbedding

        self.config.provider = "HuggingFace"
        self.config.model = multilingual_model

        hf_token = os.getenv("HUGGINGFACE_API_KEY", "")
        embed_model = HuggingFaceEmbedding(model_name=self.config.model, token=hf_token)

        # Конфигурация для пайплайна
        embedding_config = PipelineCreateEmbeddingConfig_HuggingfaceApiEmbedding(
            type="HUGGINGFACE_API_EMBEDDING",
            component=embed_model,
        )
        self.config = embedding_config
        self.handle_completion(self.config)