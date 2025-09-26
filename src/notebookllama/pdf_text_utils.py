import fitz
import re

# Функция для извлечения чистого текста из PDF через PyMuPDF


def extract_clean_text_from_pdf(file_path: str, mode: str = None, preview: bool = False) -> str:
    """
    mode: None — авто, либо "text", "blocks", "html". preview=True — выводит предпросмотр для каждого режима.
    """
    import fitz, re
    modes = ["text", "blocks", "html"] if mode is None else [mode]
    results = {}
    with fitz.open(file_path) as doc:
        for m in modes:
            text = ""
            for page in doc:
                if m == "blocks":
                    blocks = page.get_text("blocks")
                    page_text = "\n".join([b[4] for b in blocks if isinstance(b[4], str)])
                elif m == "html":
                    page_text = page.get_text("html")
                else:
                    page_text = page.get_text()
                page_text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', page_text)
                page_text = re.sub(r'\s+', ' ', page_text)
                lines = [l for l in page_text.split('\n') if len(l.strip()) > 10 and re.search(r'[а-яА-Яa-zA-Z]', l)]
                text += "\n".join(lines) + "\n"
            results[m] = text.strip()
    if preview:
        print("--- PDF text extraction preview ---")
        for m, t in results.items():
            print(f"Mode: {m}\n{text_preview(t)}\n{'-'*40}")
    # Автоматический выбор: берём режим с максимальным количеством осмысленных строк
    best_mode = max(results, key=lambda k: len(results[k].split('\n')))
    return results[best_mode]

def text_preview(text: str, lines: int = 10) -> str:
    """Показывает первые N строк текста для предпросмотра"""
    return '\n'.join(text.split('\n')[:lines])
