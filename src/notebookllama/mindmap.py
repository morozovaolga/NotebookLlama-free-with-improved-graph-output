import uuid
import os
import warnings
import json
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import List, Union

from pyvis.network import Network


class Node(BaseModel):
    id: str
    content: str


class Edge(BaseModel):
    from_id: str
    to_id: str


class MindMap(BaseModel):
    nodes: List[Node] = Field(
        description="List of nodes in the mind map, each represented as a Node object with an 'id' and concise 'content' (no more than 5 words).",
    )
    edges: List[Edge] = Field(
        description="The edges connecting the nodes of the mind map, as a list of Edge objects with from_id and to_id fields representing the source and target node IDs.",
    )

    @model_validator(mode="after")
    def validate_mind_map(self) -> Self:
        all_nodes = [el.id for el in self.nodes]
        all_edges = [el.from_id for el in self.edges] + [el.to_id for el in self.edges]
        if set(all_nodes).issubset(set(all_edges)) and set(all_nodes) != set(all_edges):
            raise ValueError(
                "There are non-existing nodes listed as source or target in the edges"
            )
        return self


# Определяем класс предупреждения
class MindMapCreationFailedWarning(UserWarning):
    """A warning returned if the mind map creation failed"""
    pass


async def get_mind_map(summary: str, highlights: List[str]) -> Union[str, None]:
    try:
        from transformers import pipeline
    except ImportError:
        warnings.warn(
            message="transformers не установлен. Установите 'pip install transformers' для генерации майндмэпа через HuggingFace.",
            category=MindMapCreationFailedWarning,
        )
        return None

    try:
        prompt = (
            f"Сгенерируй JSON-майндмэп по этому тексту на русском языке.\n"
            f"Summary: {summary}\n"
            f"Key points:\n- " + "\n- ".join(highlights) +
            "\nФормат вывода:\n"
            "{ \"nodes\": [{\"id\": \"A\", \"content\": \"...\"}, ...], \"edges\": [{\"from_id\": \"A\", \"to_id\": \"B\"}, ...] }"
        )
        print("[MindMap] Prompt для генерации:", prompt)
        
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        result = generator(prompt, max_length=512)
        print("[MindMap] Результат модели:", result)
        
        generated = result[0].get('generated_text', result[0].get('text', ''))
        json_start = generated.find('{')
        
        if json_start == -1:
            print("[MindMap][Ошибка] Модель google/flan-t5-base не сгенерировала JSON. Ответ:", generated)
            warnings.warn(
                message="Модель google/flan-t5-base не сгенерировала JSON. Попробуйте другой prompt или модель.",
                category=MindMapCreationFailedWarning,
            )
            return None
            
        json_str = generated[json_start:]
        print("[MindMap] Извлечённый JSON:", json_str)
        
        try:
            mindmap = json.loads(json_str)
        except Exception as e:
            print(f"[MindMap][Ошибка] Ошибка парсинга JSON: {e}\nJSON строка: {json_str}")
            warnings.warn(
                message=f"Ошибка парсинга JSON из ответа модели: {e}",
                category=MindMapCreationFailedWarning,
            )
            return None
            
        net = Network(directed=True, height="750px", width="100%")
        net.set_options("""
            var options = {
            "physics": {
                "enabled": false
            }
            }
            """)
            
        nodes = mindmap.get("nodes", [])
        edges = mindmap.get("edges", [])
        
        for node in nodes:
            net.add_node(n_id=node["id"], label=node["content"])
        for edge in edges:
            net.add_edge(source=edge["from_id"], to=edge["to_id"])
            
        name = str(uuid.uuid4())
        net.save_graph(name + ".html")
        print(f"[MindMap] Сохранён файл: {name}.html")
        return name + ".html"
        
    except Exception as e:
        print(f"[MindMap][Ошибка] Ошибка генерации майндмэпа через HuggingFace: {e}")
        warnings.warn(
            message=f"Ошибка генерации майндмэпа через HuggingFace: {e}",
            category=MindMapCreationFailedWarning,
        )
        return None