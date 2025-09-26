
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
