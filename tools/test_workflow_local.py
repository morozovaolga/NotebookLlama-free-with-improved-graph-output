import asyncio
import json
import os
import sys

# Ensure package imports from src (so running this script from repo root works)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from notebookllama.workflow import NotebookLMWorkflow, FileInputEvent
import notebookllama.processing as processing

# Monkeypatch parse_file to return our test text
async def fake_parse_file(path):
    text = (
        "Признаки проекта без хозяина и их последствия. Проект без хозяина — это управленческая пустота."
        "\n\n1. Решения принимаются в другом месте. Формальные руководители не могут влиять на ключевые параметры."
    )
    return text, [], []

processing.parse_file = fake_parse_file

# Monkeypatch requests.post used in workflow
import types
class FakeResponse:
    def __init__(self, body_bytes_list):
        self._lines = body_bytes_list
        self.text = b"".join(body_bytes_list).decode('utf-8', errors='replace')
    def iter_lines(self):
        for b in self._lines:
            yield b

import requests
orig_post = requests.post

def fake_post(url, json=None, stream=False, timeout=None):
    # Return a fake streaming response with a single JSON payload
    payload_str = '{"summary":"Тестовое резюме из заглушки: проект без хозяина теряет управляемость и ответственность.", "bullet_points":["Отсутствие владельца","Размытая ответственность"], "questions":["Кто отвечает за проект?"], "answers":["Никто не отвечает"], "mindmap":{"type":"Иерархическая","nodes":[{"id":"n1","label":"Проблема"}],"edges":[]}}'
    payload = payload_str.encode('utf-8')
    return FakeResponse([payload])

requests.post = fake_post

async def run_test():
    wf = NotebookLMWorkflow()
    ev = FileInputEvent(file='dummy.txt')
    # minimal context with required method
    class Ctx:
        def write_event_to_stream(self, ev):
            pass
    out = await wf.extract_file_data(ev, Ctx())
    # Print results
    # simple prints
    print('SUMMARY:', out.summary)
    print('BULLETS:', out.highlights)
    print('QUESTIONS:', out.questions)
    print('ANSWERS:', out.answers)
    print('MINDMAP:', out.mind_map)
    print('RAW_PREVIEW:', out.raw_preview)
    print('REPAIR_RAW:', out.repair_raw)
    print('FALLBACK_RAW:', out.fallback_raw)
    print('EXPAND_RAW:', out.expand_raw)

    # additional diagnostics: hex/byte dump of the raw preview and a test string
    def hex_preview(s, n=80):
        try:
            if s is None:
                return '[none]'
            if isinstance(s, str):
                b = s.encode('utf-8', errors='replace')
            else:
                b = bytes(s)
            return ' '.join(f"{c:02x}" for c in b[:n]) + (" ..." if len(b) > n else "")
        except Exception:
            return '[unprintable]'

    print('RAW_PREVIEW bytes:', hex_preview(out.raw_preview, 120))
    # test print and bytes for a sample Cyrillic word
    test_word = 'Привет'
    print("Test print of 'Привет':", test_word)
    print("Test 'Привет' bytes:", hex_preview(test_word, 40))

if __name__ == '__main__':
    asyncio.run(run_test())
