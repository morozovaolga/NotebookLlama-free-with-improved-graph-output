
import os
import pandas as pd
import json
import warnings
from datetime import datetime
from typing_extensions import override
from typing import List, Tuple, Union, Optional, Dict

# Импорт локального markdown-анализатора
from mrkdwn_analysis import MarkdownAnalyzer
from mrkdwn_analysis.markdown_analyzer import InlineParser, MarkdownParser


class MarkdownTextAnalyzer(MarkdownAnalyzer):
    @override
    def __init__(self, text: str):
        self.text = text
        parser = MarkdownParser(self.text)
        self.tokens = parser.parse()
        self.references = parser.references
        self.footnotes = parser.footnotes
        self.inline_parser = InlineParser(
            references=self.references, footnotes=self.footnotes
        )
        self._parse_inline_tokens()


def md_table_to_pd_dataframe(md_table: Dict[str, list]) -> Optional[pd.DataFrame]:
    try:
        df = pd.DataFrame()
        for i in range(len(md_table["header"])):
            ls = [row[i] for row in md_table["rows"]]
            df[md_table["header"][i]] = ls
        return df
    except Exception as e:
        warnings.warn(f"Skipping table as an error occurred: {e}")
        return None


def rename_and_remove_past_images(path: str = "static/") -> List[str]:
    renamed = []
    if os.path.exists(path) and len(os.listdir(path)) >= 0:
        for image_file in os.listdir(path):
            image_path = os.path.join(path, image_file)
            if os.path.isfile(image_path) and "_at_" not in image_path:
                with open(image_path, "rb") as img:
                    bts = img.read()
                new_path = (
                    os.path.splitext(image_path)[0].replace("_current", "")
                    + f"_at_{datetime.now().strftime('%Y_%d_%m_%H_%M_%S_%f')[:-3]}.png"
                )
                with open(
                    new_path,
                    "wb",
                ) as img_tw:
                    img_tw.write(bts)
                renamed.append(new_path)
                os.remove(image_path)
    return renamed


def rename_and_remove_current_images(images: List[str]) -> List[str]:
    imgs = []
    for image in images:
        with open(image, "rb") as rb:
            bts = rb.read()
        with open(os.path.splitext(image)[0] + "_current.png", "wb") as wb:
            wb.write(bts)
        imgs.append(os.path.splitext(image)[0] + "_current.png")
        os.remove(image)
    return imgs


async def parse_file(
    file_path: str, with_images: bool = False, with_tables: bool = False
) -> Union[Tuple[Optional[str], Optional[List[str]], Optional[List[pd.DataFrame]]]]:
    images: Optional[List[str]] = None
    text: Optional[str] = None
    tables: Optional[List[pd.DataFrame]] = None
    try:
        text = None
        if file_path.lower().endswith('.pdf'):
            # Try multiple PDF parsing backends in order of preference.
            tried = []
            parsed = False
            # 1) pdf_text_utils (project helper)
            try:
                from pdf_text_utils import extract_clean_text_from_pdf
                text = extract_clean_text_from_pdf(file_path)
                parsed = True
            except Exception as e:
                tried.append(('pdf_text_utils', str(e)))
            # 2) pdfplumber
            if not parsed:
                try:
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    parsed = True
                except Exception as e:
                    tried.append(('pdfplumber', str(e)))
            # 3) PyPDF2
            if not parsed:
                try:
                    import PyPDF2
                    text = ""
                    with open(file_path, 'rb') as fh:
                        reader = PyPDF2.PdfReader(fh)
                        for page in reader.pages:
                            try:
                                page_text = page.extract_text() or ''
                                text += page_text + "\n"
                            except Exception:
                                continue
                    parsed = True
                except Exception as e:
                    tried.append(('PyPDF2', str(e)))
            # 4) PyMuPDF (fitz)
            if not parsed:
                try:
                    import fitz
                    text = ""
                    doc = fitz.open(file_path)
                    for page in doc:
                        try:
                            text += page.get_text() + "\n"
                        except Exception:
                            continue
                    parsed = True
                except Exception as e:
                    tried.append(('fitz', str(e)))
            if not parsed:
                # report the reasons tried
                raise Exception(f"Could not parse PDF; attempted: {tried}")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        if with_images:
            rename_and_remove_past_images()
            # Здесь можно добавить обработку изображений через Pillow/pytesseract
            images = []  # Заглушка
        if with_tables and text is not None:
            analyzer = MarkdownTextAnalyzer(text)
            md_tables = analyzer.identify_tables()["Table"]
            tables = []
            for md_table in md_tables:
                table = md_table_to_pd_dataframe(md_table=md_table)
                if table is not None:
                    tables.append(table)
                    os.makedirs("data/extracted_tables/", exist_ok=True)
                    table.to_csv(
                        f"data/extracted_tables/table_{datetime.now().strftime('%Y_%d_%m_%H_%M_%S_%f')[:-3]}.csv",
                        index=False,
                    )
        else:
            tables = None
        return text, images, tables
    except Exception as e:
        warnings.warn(f"Ошибка при парсинге файла: {e}")
        return None, None, None


async def process_file(
    filename: str,
) -> Union[Tuple[str, None], Tuple[None, None], Tuple[str, str]]:
    try:
        # Локальная обработка файла: парсинг и извлечение текста
        text, _, _ = await parse_file(file_path=filename)
        if text is None:
            return None, f"Ошибка: не удалось распарсить файл {filename}"
        # Здесь можно добавить свою логику извлечения данных из текста
        # Например, анализ через HuggingFace, регулярные выражения и т.д.
        extraction_output = {"data": "(Здесь будут извлечённые данные)"}
        return json.dumps(extraction_output, indent=4), text
    except Exception as e:
        warnings.warn(f"Ошибка при обработке файла: {e}")
        return None, f"Ошибка при обработке файла: {e}"


async def get_plots_and_tables(
    file_path: str,
) -> Union[Tuple[Optional[List[str]], Optional[List[pd.DataFrame]]]]:
    _, images, tables = await parse_file(
        file_path=file_path, with_images=True, with_tables=True
    )
    return images, tables


def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """Simple extractive summary based on TF-like scoring using sentence term frequencies.

    This is intentionally lightweight and has no external dependencies. It:
    - splits text into sentences (simple regex),
    - tokenizes on word characters, lowercases,
    - computes term frequencies (TF) and scores sentences by sum of TF for tokens in the sentence,
    - returns top N sentences in the original order joined as a short summary.

    Not as good as transformer generation, but fast on CPU and suitable as a fallback.
    """
    try:
        import re
        from collections import Counter, defaultdict

        if not text or not isinstance(text, str):
            return ""

        # Simple sentence splitter (covers common cases)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) == 0:
            return ""
        if len(sentences) <= num_sentences:
            # short text — return as-is
            return "\n\n".join(sentences)

        # Tokenize words
        words = re.findall(r"\w+", text.lower())
        if not words:
            return "\n\n".join(sentences[:num_sentences])

        tf = Counter(words)

        # Score each sentence
        sent_scores = []
        for idx, s in enumerate(sentences):
            toks = re.findall(r"\w+", s.lower())
            if not toks:
                score = 0.0
            else:
                score = sum(tf.get(t, 0) for t in toks) / (len(toks) + 0.001)
            sent_scores.append((idx, score, s))

        # pick top-N by score
        top = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
        # restore original order
        top_sorted = sorted(top, key=lambda x: x[0])
        summary_sentences = [s for (_, _, s) in top_sorted]
        return "\n\n".join(summary_sentences)
    except Exception:
        # on any failure, return the first N sentences as a safe fallback
        try:
            import re

            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            return "\n\n".join(sentences[:num_sentences])
        except Exception:
            return ""


def compact_bullets(text: str, num_bullets: int = 6) -> list:
    """Return a list of compact bullet lines extracted from text using the same TF scoring.

    Fast, no external deps. Picks top-scoring sentences and returns them as short bullets.
    """
    try:
        import re
        from collections import Counter

        if not text or not isinstance(text, str):
            return []

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return []

        words = re.findall(r"\w+", text.lower())
        if not words:
            # fallback: take first num_bullets sentences
            return sentences[:num_bullets]

        tf = Counter(words)
        sent_scores = []
        for idx, s in enumerate(sentences):
            toks = re.findall(r"\w+", s.lower())
            if not toks:
                score = 0.0
            else:
                score = sum(tf.get(t, 0) for t in toks) / (len(toks) + 0.001)
            sent_scores.append((idx, score, s))

        top = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:num_bullets]
        # return as short strings, original order
        top_sorted = sorted(top, key=lambda x: x[0])
        bullets = [t[2].strip() for t in top_sorted]
        # shorten overly long bullets
        bullets_short = [ (b if len(b) <= 200 else b[:197] + '...') for b in bullets ]
        return bullets_short
    except Exception:
        try:
            import re
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            return sentences[:num_bullets]
        except Exception:
            return []


def topic_segmentation(text: str, max_chars: int = 2000) -> list:
    """Very lightweight topic segmentation: split text by paragraphs/headings into segments up to max_chars.

    Heuristics:
    - split on double newlines into paragraphs
    - start new segment when accumulated length exceeds max_chars
    - start new segment when a paragraph looks like a heading (contains 'ЭТАП' or mostly uppercase)
    """
    try:
        import re
        if not text or not isinstance(text, str):
            return []
        paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        segments = []
        cur = ''
        for p in paras:
            # detect heading-like paragraph
            is_heading = False
            up_chars = sum(1 for ch in p if ch.isupper())
            alpha_chars = sum(1 for ch in p if ch.isalpha())
            if 'ЭТАП' in p or (alpha_chars > 0 and up_chars / (alpha_chars + 0.001) > 0.6 and len(p) > 20):
                is_heading = True

            if is_heading and cur:
                segments.append(cur.strip())
                cur = p + '\n\n'
                continue

            if len(cur) + len(p) > max_chars and cur:
                segments.append(cur.strip())
                cur = p + '\n\n'
            else:
                cur += p + '\n\n'

        if cur.strip():
            segments.append(cur.strip())
        return segments
    except Exception:
        return [text]


def shorten_mindmap_nodes(mm: Union[dict, str], max_label_len: int = 120) -> Union[dict, str]:
    """If mindmap is a dict with nodes, shorten long node labels using extractive_summary or truncation."""
    try:
        if isinstance(mm, str):
            # try evaluate to dict
            import ast
            try:
                mm_parsed = ast.literal_eval(mm)
            except Exception:
                return mm
        else:
            mm_parsed = mm

        if not isinstance(mm_parsed, dict):
            return mm

        nodes = mm_parsed.get('nodes', [])
        for n in nodes:
            if isinstance(n, dict) and 'label' in n and isinstance(n['label'], str):
                lab = n['label']
                if len(lab) > max_label_len:
                    # Try to create a single-sentence extractive summary
                    s = extractive_summary(lab, num_sentences=1)
                    if s and len(s) < len(lab):
                        n['label'] = s if len(s) <= max_label_len else s[:max_label_len-3] + '...'
                    else:
                        n['label'] = lab[:max_label_len-3] + '...'
        mm_parsed['nodes'] = nodes
        return mm_parsed
    except Exception:
        return mm
