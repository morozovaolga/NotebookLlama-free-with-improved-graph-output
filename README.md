
# NotebookLlaMa🦙

## Форк популярного проекта [Llama](https://github.com/run-llama/notebookllama)

Этот репозиторий — мой кастомизированный форк проекта Llama. Я решила сделать его полностью бесплатным, убрать все платные зависимости и улучшить вывод майндмэпов, диаграмм и графиков.

**Основные отличия:**
- Бесплатное использование (только бесплатные API и модели)
- Улучшенный вывод майндмэпов, графиков и диаграмм
- Открытый код для самостоятельной доработки

Исходный проект: [github.com/run-llama/notebookllama](https://github.com/run-llama/notebookllama)

---


### Требования

This project uses `uv` to manage dependencies. Before you begin, make sure you have `uv` installed.

On macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more install options, see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

---


### Быстрый старт


**1. Клонируйте репозиторий**

```bash
git clone https://github.com/run-llama/notebookllama
cd notebookllama/
```


**2. Установите зависимости**

```bash
uv sync
```


**3. Укажите API-ключи**

First, create your `.env` file by renaming the example file:

```bash
mv .env.example .env
```

>
> - For **North America**: This is the default region - no configuration necesary.
> - For **Europe (EU)**: Uncomment and set `LLAMACLOUD_REGION="eu"`

Далее откройте файл `.env` и добавьте ваши ключи:

- `HUGGINGFACE_API_KEY`: получите на [HuggingFace](https://huggingface.co/settings/tokens)
- `LLAMACLOUD_API_KEY`: получите на [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

> **🌍 Региональная поддержка**: LlamaCloud работает в нескольких регионах. Для Европы укажите:
> - Для **Европы (EU)**: `LLAMACLOUD_REGION="eu"`


**4. Активируйте виртуальное окружение**

(on mac/unix)

```bash
source .venv/bin/activate
```

(on Windows):

```bash
.\.venv\Scripts\activate
```


**5. Создайте агента и пайплайн LlamaCloud**

You will now execute two scripts to configure your backend agents and pipelines.

First, create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

Next, run the interactive setup wizard to configure your index pipeline.


> **⚡ Быстрый старт (по умолчанию HuggingFace):**
> Для быстрого запуска выберите **"With Default Settings"** — будет создан пайплайн с бесплатной моделью HuggingFace (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`).

> **🧠 Расширенный режим (свои модели):**
> Для выбора другой бесплатной модели HuggingFace (например, `all-MiniLM-L6-v2`, `bge-small-en-v1.5`), выберите **"With Custom Settings"** и следуйте инструкциям.

Run the wizard with the following command:

```bash
uv run tools/create_llama_cloud_index.py
```



**6. Запустите backend-сервисы**

Эта команда запустит необходимые контейнеры Postgres, Jaeger, Adminer и Ollama:

```bash
docker compose up -d
```

**7. Укажите параметры моделей и задач**

В файле `config.yaml` (или `.env`) укажите нужные параметры для LLM и задач:

```yaml
llm:
	provider: ollama
	model: mistral
	endpoint: http://localhost:11434
	max_new_tokens: 1024
	temperature: 0.1
	top_p: 0.9

tasks:
	summary:
		model: mistral
	mindmap:
		model: mistral
	facts:
		model: mistral
```

**8. Запустите приложение**

Сначала запустите сервер:

```powershell
# Recommended: run as a module so package imports work (ensures `src` is on PYTHONPATH)
$env:PYTHONPATH = "src"; python -m notebookllama.server
```

If you prefer the simpler script form (may require adjusting PYTHONPATH), you can run:

```powershell
# Make sure `src` is on PYTHONPATH first, or run with the module form above
$env:PYTHONPATH = "src"; python src\notebookllama\server.py
```

В новом терминале запустите Streamlit:

```powershell
python -m streamlit run src/notebookllama/Home.py
```

**Остановка Streamlit (Windows PowerShell):**
Если нужно завершить работу Streamlit, используйте команду:

```powershell
Stop-Process -Name streamlit -Force
```
или просто закройте окно терминала, либо нажмите `Ctrl+C`.

> [!IMPORTANT]
>
> _You might need to install `ffmpeg` if you do not have it installed already_

Note on PDF parsing: for more reliable PDF extraction the project prefers `pdfplumber`.
If you experience truncated or failed PDF parsing, install it in your environment:

```powershell
pip install pdfplumber
```


Откройте приложение в браузере: `http://localhost:8501/`

---


### Вклад и доработка

Contribute to this project following the [guidelines](./CONTRIBUTING.md).


### Лицензия

This project is provided under an [MIT License](./LICENSE).
