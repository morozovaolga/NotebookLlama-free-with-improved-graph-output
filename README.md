
# NotebookLlaMa🦙

Форк проекта notebookllama с упором на локальный запуск (без платных облачных LLM по умолчанию), улучшенный вывод майндмэпов и быстрый hybrid‑режим.

Исходный проект: [github.com/run-llama/notebookllama](https://github.com/run-llama/notebookllama)

---

## Требования

- Python 3.10+
- Рекомендуется создать виртуальное окружение и установить зависимости (`pip install -r requirements.txt`).
- Для более надёжного извлечения текста из PDF установите `pdfplumber` (опционально).

На macOS / Linux можно установить `uv` по инструкции проекта; для Windows следуйте официальной документации `uv`.

---

## Быстрый старт (PowerShell)

1. Создайте и активируйте виртуальное окружение, затем установите зависимости:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. При желании создайте файл `.env` на основе примера:

```powershell
Copy-Item .env.example .env
```

1. Опционально: предзагрузите агенты/индекс (если используете LlamaCloud):

```powershell
uv run tools/create_llama_extract_agent.py
uv run tools/create_llama_cloud_index.py
```

1. Запустите локальные сервисы (если требуется):

```powershell
docker compose up -d
```

1. Запустите Streamlit (рекомендуется запускать как модуль):

```powershell
$env:PYTHONPATH = 'src'; python -m streamlit run src/notebookllama/Home.py
```

Откройте [http://localhost:8501/](http://localhost:8501/) в браузере.

---

## Hybrid режим: быстрый extractive + фоновый LLM‑рефайн

Режим "hybrid" показывает мгновенный extractive‑результат (локальный, быстрый) и одновременно запускает фоновый LLM‑рефайн для более структурированного и подробного вывода.

Основные элементы:

- UI-кнопка: "Extractive + LLM refine" — показывает extractive сразу и запускает рефайн в фоне.
- Sidebar опция: "Always use extractive by default (fast)" — при включении обычная обработка будет выполнять только extractive путь.
- Sidebar опция: "Auto-apply LLM refine result when ready" — при включении refined результат автоматически заменит extractive вывод; иначе появится превью с кнопками "Apply" / "Dismiss".

Техническая заметка: фоновая задача записывает результат в временный JSON‑файл; основной поток UI периодически проверяет этот файл и аккуратно применяет результат в session_state — это снижает риск гонок между потоками.

---

## Sidebar — HF fallback и параметры локального генератора

Панель слева позволяет настроить поведение fallback‑модели HuggingFace (если Ollama недоступен), выбрать модель, предзагрузить её и задать параметры генерации (temperature, max_tokens, top_k, top_p, do_sample).

Рекомендация для быстрых CPU‑тестов: модель `distilgpt2`, `temperature=0.0`, `max_tokens=256`, `do_sample=false`.

---

## Примечания по конфигурации

- Параметры LLM/тасков можно указать в `config.yaml` или через переменные окружения.
- Если используете облачные сервисы, добавьте соответствующие ключи в `.env` (`HUGGINGFACE_API_KEY`, `LLAMACLOUD_API_KEY` и т.д.).

---

## Лицензия

Проект распространяется под лицензией MIT. Подробнее: ./LICENSE
