import uuid
import os
import warnings
import json
import re
import requests
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


class MindMapCreationFailedWarning(UserWarning):
    """A warning returned if the mind map creation failed"""
    pass


def create_fallback_mindmap(summary: str, highlights: List[str]) -> dict:
    """Создает простой майндмэп как fallback"""
    nodes = [{"id": "A", "content": "Главная тема"}]
    edges = []
    
    # Добавляем узлы из highlights
    if highlights:
        for i, highlight in enumerate(highlights[:5]):  # Ограничиваем до 5
            node_id = chr(ord('B') + i)
            # Обрезаем текст до 30 символов
            content = highlight[:30] + "..." if len(highlight) > 30 else highlight
            nodes.append({"id": node_id, "content": content})
            edges.append({"from_id": "A", "to_id": node_id})
    
    # Если нет highlights, используем summary
    elif summary:
        # Разбиваем summary на предложения
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        for i, sentence in enumerate(sentences[:4]):  # Первые 4 предложения
            node_id = chr(ord('B') + i)
            # Обрезаем до 40 символов
            content = sentence[:40] + "..." if len(sentence) > 40 else sentence
            nodes.append({"id": node_id, "content": content})
            edges.append({"from_id": "A", "to_id": node_id})
    
    # Если совсем ничего нет
    else:
        nodes.extend([
            {"id": "B", "content": "Документ обработан"},
            {"id": "C", "content": "Нет извлеченных данных"}
        ])
        edges.extend([
            {"from_id": "A", "to_id": "B"},
            {"from_id": "A", "to_id": "C"}
        ])
    
    return {"nodes": nodes, "edges": edges}


def extract_json_from_text(text: str) -> dict:
    """Пытается извлечь JSON из текста различными способами"""
    # Способ 1: Найти JSON блок
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    # Способ 2: Найти массивы nodes и edges отдельно
    nodes_pattern = r'"nodes"\s*:\s*\[(.*?)\]'
    edges_pattern = r'"edges"\s*:\s*\[(.*?)\]'
    
    nodes_match = re.search(nodes_pattern, text, re.DOTALL)
    edges_match = re.search(edges_pattern, text, re.DOTALL)
    
    if nodes_match and edges_match:
        try:
            nodes_text = '[' + nodes_match.group(1) + ']'
            edges_text = '[' + edges_match.group(1) + ']'
            nodes = json.loads(nodes_text)
            edges = json.loads(edges_text)
            return {"nodes": nodes, "edges": edges}
        except:
            pass
    
    return None


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
        key_points = "\n- ".join(highlights) if highlights else "Нет ключевых точек"
        prompt = f"""Создай JSON майндмэп:
Тема: {summary[:100]}
Ключевые точки: {key_points[:200]}

Формат ответа (ТОЛЬКО JSON):
{{"nodes": [{{"id": "A", "content": "Главная тема"}}, {{"id": "B", "content": "Пункт 1"}}, {{"id": "C", "content": "Пункт 2"}}], "edges": [{{"from_id": "A", "to_id": "B"}}, {{"from_id": "A", "to_id": "C"}}]}}"""

        print("[MindMap] Prompt для генерации:", prompt[:200] + "...")

        models_to_try = [
            "ollama",  # сначала пробуем Ollama
            "google/flan-t5-small",
            "google/flan-t5-base"
        ]

        mindmap_data = None

        for model_name in models_to_try:
            try:
                print(f"[MindMap] Пробуем модель: {model_name}")

                if model_name == "ollama":
                    # Генерация через Ollama API
                    ollama_prompt = prompt
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "mistral", "prompt": ollama_prompt},
                        stream=True
                    )
                    generated = ""
                    def safe_decode(b: bytes) -> str:
                        try:
                            return b.decode("utf-8")
                        except Exception:
                            try:
                                return b.decode("latin-1")
                            except Exception:
                                try:
                                    return b.decode("utf-8", errors="replace")
                                except Exception:
                                    return ""

                    for line in response.iter_lines():
                        if not line:
                            continue
                        dec = safe_decode(line)
                        try:
                            data = json.loads(dec)
                            generated += data.get("response", "")
                        except Exception:
                            generated += dec
                    print(f"[DEBUG] Ollama response: {generated[:200]}...")
                elif "flan-t5" in model_name:
                    generator = pipeline("text2text-generation", model=model_name)
                    result = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
                    print(f"[DEBUG] LlamaCloud response: {result}")
                    generated = result[0].get('generated_text', '')
                else:
                    generator = pipeline("text-generation", model=model_name)
                    result = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
                    print(f"[DEBUG] LlamaCloud response: {result}")
                    generated = result[0].get('generated_text', '').replace(prompt, '')

                print(f"[MindMap] Результат модели {model_name}:", generated[:200] + "...")

                mindmap_data = extract_json_from_text(generated)

                if mindmap_data and 'nodes' in mindmap_data and 'edges' in mindmap_data:
                    print(f"[MindMap] Успешно извлечен JSON из модели {model_name}")
                    break

            except Exception as e:
                print(f"[MindMap] Ошибка с моделью {model_name}: {e}")
                continue

        if not mindmap_data:
            print("[MindMap] Используем fallback майндмэп")
            mindmap_data = create_fallback_mindmap(summary, highlights)

        net = Network(directed=True, height="750px", width="100%")
        net.set_options("""
            var options = {
                "physics": {
                    "enabled": true,
                    "stabilization": {"iterations": 100}
                },
                "nodes": {
                    "color": {"background": "#97C2FC", "border": "#2B7CE9"},
                    "font": {"size": 14}
                },
                "edges": {
                    "color": {"color": "#848484"},
                    "arrows": {"to": {"enabled": true, "scaleFactor": 1.2}}
                }
            }
            """)

        nodes = mindmap_data.get("nodes", [])
        edges = mindmap_data.get("edges", [])

        for node in nodes:
            node_id = node.get("id", str(len(net.nodes)))
            content = node.get("content", "Узел")[:50]
            net.add_node(n_id=node_id, label=content, title=content)

        for edge in edges:
            from_id = edge.get("from_id")
            to_id = edge.get("to_id")
            if from_id and to_id:
                net.add_edge(source=from_id, to=to_id)

        name = str(uuid.uuid4())
        file_path = name + ".html"
        net.save_graph(file_path)
        print(f"[MindMap] Сохранён файл: {file_path}")
        return file_path

    except Exception as e:
        print(f"[MindMap][Ошибка] Общая ошибка генерации майндмэпа: {e}")

        try:
            fallback_data = create_fallback_mindmap(summary, highlights)
            net = Network(directed=True, height="750px", width="100%")

            for node in fallback_data["nodes"]:
                net.add_node(n_id=node["id"], label=node["content"])
            for edge in fallback_data["edges"]:
                net.add_edge(source=edge["from_id"], to=edge["to_id"])

            name = str(uuid.uuid4())
            file_path = name + ".html"
            net.save_graph(file_path)
            print(f"[MindMap] Сохранён fallback файл: {file_path}")
            return file_path

        except Exception as fallback_error:
            print(f"[MindMap][Критическая ошибка] Не удалось создать даже fallback майндмэп: {fallback_error}")
            warnings.warn(
                message=f"Критическая ошибка генерации майндмэпа: {e}",
                category=MindMapCreationFailedWarning,
            )
            return None