import sys, os, inspect, asyncio

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from notebookllama.processing import parse_file

test_md = os.path.abspath(os.path.join(ROOT, "data", "test", "md_sample.md"))
print("Test file:", test_md)
if not os.path.exists(test_md):
    print("Файл не найден:", test_md)
    raise SystemExit(1)

try:
    res = parse_file(test_md)
    if inspect.iscoroutine(res):
        res = asyncio.run(res)
    # parse_file может вернуть (md_text, images, tables)
    if isinstance(res, tuple) and len(res) >= 1:
        md_text = res[0]
    else:
        md_text = res
    if md_text is None:
        print("parse_file вернул None для текста")
    else:
        print("Extracted text length:", len(md_text))
        print("Preview:\n", md_text[:1000])
except Exception as e:
    print("Ошибка при parse_file:", repr(e))