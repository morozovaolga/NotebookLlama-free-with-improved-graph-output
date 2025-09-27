import sys
import os
import pytest

# Ensure `src` is on PYTHONPATH for tests run from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from notebookllama.processing import extractive_summary, compact_bullets


def test_extractive_summary_basic():
    text = (
        "Это первый абзац. Здесь важная мысль.\n\n"
        "Второй абзац содержит дополнительные детали и примеры. Еще одно предложение.\n\n"
        "Третий абзац с завершающей мыслью."
    )
    s = extractive_summary(text, num_sentences=2)
    assert isinstance(s, str)
    assert len(s) > 0


def test_compact_bullets_basic():
    text = "Первая мысль. Вторая мысль. Третья мысль."
    bullets = compact_bullets(text, num_bullets=3)
    assert isinstance(bullets, list) or isinstance(bullets, str)