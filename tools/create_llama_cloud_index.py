import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from dotenv import load_dotenv
from cli.embedding_app import EmbeddingSetupApp
from src.notebookllama.utils import create_llamacloud_client

from llama_cloud import (
    PipelineTransformConfig_Advanced,
    AdvancedModeTransformConfigChunkingConfig_Sentence,
    AdvancedModeTransformConfigSegmentationConfig_Page,
    PipelineCreate,
)


def main():
    """
    Create a new Llama Cloud index with the given embedding configuration.
    """
    load_dotenv()
    client = create_llamacloud_client()

    app = EmbeddingSetupApp()
    try:
        embedding_config = app.run()
    except Exception as e:
        print(f"Ошибка при запуске EmbeddingSetupApp: {e}")
        return 1

    if embedding_config:
        try:
            segm_config = AdvancedModeTransformConfigSegmentationConfig_Page(mode="page")
            chunk_config = AdvancedModeTransformConfigChunkingConfig_Sentence(
                chunk_size=1024,
                chunk_overlap=200,
                separator="<whitespace>",
                paragraph_separator="\n\n\n",
                mode="sentence",
            )

            transform_config = PipelineTransformConfig_Advanced(
                segmentation_config=segm_config,
                chunking_config=chunk_config,
                mode="advanced",
            )

            pipeline_request = PipelineCreate(
                name="notebooklm_pipeline",
                embedding_config=embedding_config,
                transform_config=transform_config,
            )

            try:
                pipeline = asyncio.run(
                    client.pipelines.upsert_pipeline(request=pipeline_request)
                )
            except Exception as e:
                print(f"Ошибка при создании пайплайна: {e}")
                return 1


            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
            env_path = os.path.abspath(env_path)
            print(f"Запись LLAMACLOUD_PIPELINE_ID в файл: {env_path}")
            with open(env_path, "a") as f:
                f.write(f'\nLLAMACLOUD_PIPELINE_ID="{pipeline.id}"')

            print(f"Пайплайн успешно создан! ID: {pipeline.id}")
            print("Проверьте, что LLAMACLOUD_PIPELINE_ID добавлен в .env")
            return 0
        except Exception as e:
            print(f"Ошибка при конфигурировании пайплайна: {e}")
            return 1
    else:
        print("No embedding configuration provided")
        return 1


if __name__ == "__main__":
    main()