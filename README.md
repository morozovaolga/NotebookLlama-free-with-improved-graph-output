
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
 
## Hybrid режим: быстрый extractive + фоновый LLM‑рефайн (новая функциональность)

Добавлен удобный режим "hybrid" — он даёт мгновенный локальный extractive‑результат (для интерактивной работы на CPU) и параллельно запускает полноценный LLM‑workflow для получения более точного и структурированного результата.

Ключевые элементы и поведение:

- Кнопка в UI: "Extractive + LLM refine" — при нажатии приложение сразу показывает extractive summary, bullet points и упрощённый mindmap, затем в фоне запускается полный LLM‑рефайн.
- Sidebar чекбокс: "Always use extractive by default (fast)" — если включён, то при нажатии обычной кнопки "Process Document" будет выполняться только быстрый extractive путь (удобно для итеративной работы).
- Sidebar чекбокс: "Auto-apply LLM refine result when ready" — если включён, то когда фоновой LLM‑рефайн завершится, его результат автоматически заменит отображаемый extractive‑результат. Если выключен, появится превью с кнопками "Apply refined result" и "Dismiss refined result".

Преимущества:

- Мгновенная обратная связь на CPU: можно быстро просмотреть основные тезисы и настроить документ или заголовки.
- Последующая автоматическая или ручная замена на более богатый LLM‑результат без необходимости заново запускать обработку.

Пример использования (PowerShell):

```powershell
# Запустите Streamlit как обычно
$env:NL_DEBUG='1'; python -m streamlit run src/notebookllama/Home.py

# В sidebar: можно включить "Always use extractive by default" для быстрых итераций.
# Загрузите PDF и нажмите "Extractive + LLM refine" — сразу увидите краткий summary,
# а через некоторое время подойдет refined результат (авто‑замена, если включено).
```

Замечание по безопасности/стабильности:

- Фоновый рефайн запускается в отдельном потоке и сохраняет результат в сессионном состоянии Streamlit. В редких случаях возможны гонки; если вы столкнётесь с редкими несинхронными состояниями, напишите — я помогу укрепить обмен данными (через временные файлы/очередь или дополнительную блокировку).
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
$env:PYTHONPATH = "src"; python -m notebookllama.server$
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

---

### Sidebar (левая колонка) — параметры и управление HF fallback

В `Home.py` добавлена удобная панель слева (sidebar) для локального тестирования и управления fallback-моделью HuggingFace. Ниже описание каждого элемента и рекомендаций по использованию.

- Use local HF fallback if Ollama unavailable (checkbox)
  - Включите, если хотите, чтобы при недоступности локального Ollama приложение автоматически пыталось использовать локальную модель HuggingFace (через `transformers`).
  - Рекомендуется включать только на машинах, где установлены необходимые зависимости и есть достаточно места для моделей.

- HF model (quick select) / HF fallback model (custom)
  - Быстрый селектор предлагает распространённые небольшие модели (`gpt2`, `distilgpt2`, и т.д.).
  - Поле "custom" позволяет указать любую модель HuggingFace (например `distilgpt2` или путь к локальной модели).
  - Пример: `distilgpt2` — хорош для CPU и быстрых тестов; большие модели требуют GPU и много памяти.

- Load model (button)
  - Нажмите перед обработкой документов, чтобы заранее скачать и закэшировать модель. Это значительно ускорит последующую генерацию (особенно при первом запуске).
  - Кнопка запускает фоновую задачу загрузки; статус отображается под контролами.

- Save model (button)
  - Сохраняет выбранную модель в файл `.env` (ключ `HF_FALLBACK_MODEL`) для постоянного использования.

- HF temperature (slider)
  - Параметр `temperature` управляет стохастичностью генерации (0.0 — детерминированно, 1.0 — максимум случайности).
  - Для стабильных структурированных выходов (JSON) используйте маленькие значения (0.0–0.2).

- HF max tokens (number)
  - Максимальное количество токенов, которое модель сгенерирует при HF‑fallback (от 16 до 2048).
  - Для саммари/майндмэпа обычно достаточно 128–512 токенов. Большие значения увеличивают время и память.

- HF top_k (number) и HF top_p (slider)
  - `top_k` — ограничение по количеству наиболее вероятных токенов (0 = отключено).
  - `top_p` — nucleus sampling (вероятность выборки). Вместе с `temperature` дают контроль над разнообразием генерации.
  - Для более детерминированного поведения используйте `top_k` ~ 40–100 и `top_p` ~ 0.8–0.95.

- HF do_sample (checkbox)
  - Включение `do_sample` включает стохастический сэмплинг (требуется вместе с `temperature`/`top_k`/`top_p`).
  - Если хотите максимально детерминированный вывод (лучше для JSON), оставляйте выключенным.

Рекомендации по использованию

- Для быстрых локальных тестов (на CPU): модель `distilgpt2`, `temperature=0.0`, `max_tokens=256`, `do_sample=false`.
- Перед большим прогоном предварительно нажмите "Load model" — это уменьшит задержку во время обработки документа.
- Если вы видите, что Ollama таймаутит, попробуйте включить HF‑fallback или увеличить `WORKFLOW_TIMEOUT` (env var или через UI Retry banner).

Диагностика

- В UI есть expander'ы для диагностики: "Raw LLM response preview", "Fallback (mindmap) LLM preview", "Repair LLM preview" и "Expand summary LLM preview" — используйте их, чтобы понять, почему pipeline сделал repair/fallback.
- Для подробных логов запускайте Streamlit с `NL_DEBUG=1`:
```powershell
$env:NL_DEBUG='1'; python -m streamlit run src/notebookllama/Home.py
```



### Лицензия

This project is provided under an [MIT License](./LICENSE).
# Рекомендуемые модели и пресеты параметров

Ниже — практичные пресеты для трёх задач: создание саммари (summary), генерация майндмэпов (mindmap) и подсказок/описаний для визуализаций и графиков (graphs). Для каждого пресета указаны рекомендации для трёх окружений: CPU (малые модели), Medium (средние), и GPU (большие модели). Также даны рекомендуемые значения параметров генерации.

Общие рекомендации:
- Перед большим прогоном нажмите "Load model" в sidebar, чтобы заранее скачать и закэшировать модель.
- Для строго структурированного JSON используйте маленькое `temperature` (0.0–0.2) и `do_sample=false`.
- Увеличивайте `max_tokens` для более подробного вывода, но учитывайте рост времени и памяти.

1) Summary (краткие и подробные сводки)

- CPU (легковесный):
  - Model: `distilgpt2` или `gpt2`
  - temperature: 0.0
  - max_tokens: 128
  - top_k: 50
  - top_p: 0.9
  - do_sample: false

- Medium (баланс скорости и качества):
  - Model: `facebook/opt-125m`, `EleutherAI/gpt-neo-125M` или `gpt2-medium`
  - temperature: 0.0–0.1
  - max_tokens: 256
  - top_k: 40
  - top_p: 0.95
  - do_sample: false

- GPU (детальные саммари):
  - Model: `gpt2-large`, `bigscience/bloom-3b` (требуют GPU и много памяти)
  - temperature: 0.0–0.2
  - max_tokens: 512
  - top_k: 50
  - top_p: 0.95
  - do_sample: false (можно true для креативности)

2) Mindmap (иерархическая структура)

- CPU:
  - Model: `distilgpt2`
  - temperature: 0.0
  - max_tokens: 128
  - top_k: 40
  - top_p: 0.9
  - do_sample: false

- Medium:
  - Model: `gpt2-medium` или `distilgpt2`
  - temperature: 0.0–0.15
  - max_tokens: 256
  - top_k: 50
  - top_p: 0.92
  - do_sample: false

- GPU:
  - Model: `gpt2-large`, `bloom-1b7` или другие крупные модели
  - temperature: 0.0–0.2
  - max_tokens: 512
  - top_k: 60
  - top_p: 0.95
  - do_sample: false или true для более креативных карт

3) Graphs / Visualization prompts (подписи, описания, подсказки для визуализации)

- CPU:
  - Model: `distilgpt2`
  - temperature: 0.1
  - max_tokens: 80
  - top_k: 40
  - top_p: 0.9
  - do_sample: true

- Medium:
  - Model: `gpt2-medium`, `EleutherAI/gpt-neo-125M`
  - temperature: 0.1–0.3
  - max_tokens: 120
  - top_k: 50
  - top_p: 0.92
  - do_sample: true

- GPU:
  - Model: `gpt2-large`, `bloom-3b`
  - temperature: 0.15–0.4
  - max_tokens: 160
  - top_k: 60
  - top_p: 0.95
  - do_sample: true

Примеры команд (PowerShell)

- Предзагрузка модели (предварительное кеширование весов):

```powershell
$env:HF_FALLBACK_MODEL='distilgpt2'; python tools/test_workflow_local.py --preload
```

- Запуск Streamlit с подробными логами:

```powershell
$env:NL_DEBUG='1'; python -m streamlit run src/notebookllama/Home.py
```

Если хотите, могу добавить JSON‑шаблоны prompt'ов для каждого пресета (строго JSON‑ответы), чтобы уменьшить частоту repair/fallback в workflow.

---

### Hybrid режим: быстрый extractive + фоновый LLM‑рефайн (новая функциональность)

Добавлен удобный режим "hybrid" — он даёт мгновенный локальный extractive‑результат (для интерактивной работы на CPU) и параллельно запускает полноценный LLM‑workflow для получения более точного и структурированного результата.

Ключевые элементы и поведение:
- Кнопка в UI: "Extractive + LLM refine" — при нажатии приложение сразу показывает extractive summary, bullet points и упрощённый mindmap, затем в фоне запускается полный LLM‑рефайн.
- Sidebar чекбокс: "Always use extractive by default (fast)" — если включён, то при нажатии обычной кнопки "Process Document" будет выполняться только быстрый extractive путь (удобно для итеративной работы).
- Sidebar чекбокс: "Auto-apply LLM refine result when ready" — если включён, то когда фоновой LLM‑рефайн завершится, его результат автоматически заменит отображаемый extractive‑результат. Если выключен, появится превью с кнопками "Apply refined result" и "Dismiss refined result".

Преимущества:
- Мгновенная обратная связь на CPU: можно быстро просмотреть основные тезисы и настроить документ или заголовки.
- Последующая автоматическая или ручная замена на более богатый LLM‑результат без необходимости заново запускать обработку.

Пример использования (PowerShell):

```powershell
# Запустите Streamlit как обычно
$env:NL_DEBUG='1'; python -m streamlit run src/notebookllama/Home.py

# В sidebar: можно включить "Always use extractive by default" для быстрых итераций.
# Загрузите PDF и нажмите "Extractive + LLM refine" — сразу увидите краткий summary,
# а через некоторое время подойдет refined результат (авто‑замена, если включено).
```

Замечание по безопасности/стабильности:
- Фоновый рефайн запускается в отдельном потоке и сохраняет результат в сессионном состоянии Streamlit. В редких случаях возможны гонки; если вы столкнётесь с редкими несинхронными состояниями, напишите — я помогу укрепить обмен данными (через временные файлы/очередь или дополнительную блокировку).


### Лицензия

This project is provided under an [MIT License](./LICENSE).
