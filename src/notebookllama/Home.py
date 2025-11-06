
import streamlit as st
import io
import os
import asyncio
import threading
import tempfile as temp
import hashlib
import sys
import time
import randomname
import streamlit.components.v1 as components
from pathlib import Path
import sys
import os

# Ensure project `src` is on sys.path when running with `streamlit run` or directly
try:
    import notebookllama  # type: ignore
except Exception:
    # add repository src/ to sys.path (only for developer convenience)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

# Try to ensure stdout/stderr use UTF-8 on Windows consoles to avoid garbled Cyrillic
try:
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    else:
        import io
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
    else:
        import io
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
except Exception:
    pass

from dotenv import load_dotenv

# Load environment variables from .env (if present). This makes Streamlit/Windows runs
# pick up pgql_user/pgql_psw/pgql_db without requiring manual export in the shell.
try:
    # Try to load .env with explicit encoding handling
    env_path = Path(__file__).resolve().parents[2] / '.env'
    if env_path.exists():
        try:
            # Try loading with utf-8-sig to handle BOM if present
            load_dotenv(dotenv_path=str(env_path), encoding='utf-8-sig')
        except (UnicodeDecodeError, Exception):
            # Fallback: try standard load_dotenv
            try:
                load_dotenv(dotenv_path=str(env_path))
            except Exception:
                pass
    else:
        load_dotenv()
except Exception:
    # If python-dotenv is not installed, rely on environment variables already set.
    pass

from notebookllama.documents import ManagedDocument, DocumentManager
from typing import Tuple
from notebookllama.workflow import NotebookLMWorkflow, FileInputEvent, NotebookOutputEvent

engine_url = f"postgresql+psycopg2://{os.getenv('pgql_user')}:{os.getenv('pgql_psw')}@localhost:5432/{os.getenv('pgql_db')}"
document_manager = DocumentManager(engine_url=engine_url)
WF = NotebookLMWorkflow(timeout=600)


# Warm-up Ollama in background to reduce first-request latency (non-blocking)
def _warm_up_ollama_background(endpoint: str = "http://localhost:11434"):
    try:
        import requests
        # small lightweight generate to trigger model load; ignore failures
        try:
            ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
            requests.post(endpoint.rstrip('/') + '/api/generate', json={"model": ollama_model, "prompt": "Ping"}, timeout=5)
        except Exception:
            pass
    except Exception:
        pass

# start warm-up once per session (non-blocking)
try:
    if not getattr(st.session_state, '_ollama_warmed', False):
        t = threading.Thread(target=_warm_up_ollama_background, daemon=True)
        t.start()
        st.session_state._ollama_warmed = True
except Exception:
    pass


# Read the HTML file
def read_html_file(file_path: str) -> str:
    """Read HTML file with safe encoding handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with utf-8-sig to handle BOM, then fallback to other encodings
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback: read as bytes and decode with error handling
            with open(file_path, "rb") as f:
                content = f.read()
                try:
                    return content.decode("utf-8", errors="replace")
                except Exception:
                    # Last resort: try latin-1 (never fails)
                    return content.decode("latin-1", errors="replace")


def is_ollama_available(endpoint: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    """Lightweight check whether the local Ollama HTTP API responds.
    Returns True if a short request succeeds, False otherwise.
    This is intentionally non-fatal and only used to show a UI hint.
    """
    try:
        import urllib.request, json
        # First try the /api/tags endpoint which is lightweight and always available
        try:
            tags_url = endpoint.rstrip('/') + '/api/tags'
            req = urllib.request.Request(tags_url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= getattr(resp, 'status', 200) < 400:
                    return True
        except Exception:
            pass
        
        # Fallback: try a simple GET to the root URL
        try:
            with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
                body = resp.read(1024)
                try:
                    txt = body.decode('utf-8', errors='ignore')
                except Exception:
                    txt = ''
                if 200 <= getattr(resp, 'status', 200) < 400:
                    # look for a short hint that Ollama is running
                    if 'ollama' in txt.lower() or 'running' in txt.lower():
                        return True
                    # even if body doesn't include text, a 200 response is likely OK
                    return True
        except Exception:
            pass
        
        # Last fallback: try the generate endpoint with a tiny POST
        try:
            gen_url = endpoint.rstrip('/') + '/api/generate'
            ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
            data = json.dumps({"model": ollama_model, "prompt": "ping"}).encode("utf-8")
            req = urllib.request.Request(gen_url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= getattr(resp, 'status', 200) < 400
        except Exception:
            return False
    except Exception:
        return False


async def run_workflow(
    file: io.BytesIO, document_title: str
) -> Tuple[str, str, str, str, str]:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("notebookllama")
    logger.info("[run_workflow] Начало обработки файла")
    # Create temp file with proper Windows handling
    with temp.NamedTemporaryFile(suffix=".pdf", delete=False) as fl:
        content = file.getvalue()
        fl.write(content)
        fl.flush()  # Ensure data is written
        temp_path = fl.name
    logger.info(f"[run_workflow] Временный файл создан: {temp_path}")

    try:
        st_time = int(time.time() * 1000000)
        logger.info(f"[run_workflow] Создаём FileInputEvent для файла: {temp_path}")
        ev = FileInputEvent(file=temp_path)
        logger.info("[run_workflow] Запуск WF.run...")
        result: NotebookOutputEvent = await WF.run(start_event=ev)
        logger.info("[run_workflow] WF.run завершён, формируем результаты...")

        q_and_a = ""
        for q, a in zip(result.questions, result.answers):
            q_and_a += f"**{q}**\n\n{a}\n\n"
        bullet_points = "## Bullet Points\n\n- " + "\n- ".join(result.highlights)

        mind_map = result.mind_map
        # Protect against None or non-string mind_map values before Path()
        try:
            if mind_map and Path(str(mind_map)).is_file():
                logger.info(f"[run_workflow] Майндмэп сохранён как файл: {mind_map}")
                mind_map = read_html_file(str(mind_map))
                try:
                    os.remove(str(result.mind_map))
                except OSError:
                    pass  # File might be locked on Windows
        except Exception as _e:
            # If anything goes wrong checking the mindmap file, fall back to keeping the value
            logger.debug(f"[run_workflow] mind_map check failed: {_e}")

        end_time = int(time.time() * 1000000)
        logger.info(f"[run_workflow] Документ обработан за {(end_time-st_time)/1e6:.2f} сек")
        # Ensure all document fields are strings and safe for DB insertion
        def safe_str(val):
            try:
                if isinstance(val, bytes):
                    try:
                        return val.decode('utf-8')
                    except Exception:
                        try:
                            return val.decode('latin-1')
                        except Exception:
                            return val.decode('utf-8', errors='replace')
                elif val is None:
                    return ""
                else:
                    return str(val)
            except Exception:
                return ""

        md_content_safe = safe_str(result.md_content)
        summary_safe = safe_str(result.summary)
        q_and_a_safe = safe_str(q_and_a)
        mind_map_safe = safe_str(mind_map)
        bullet_points_safe = safe_str(bullet_points)

        logger.info(f"[run_workflow] Подготавливаем документ для записи: name={document_title}, content_len={len(md_content_safe)}, summary_len={len(summary_safe)}")
        try:
            document_manager.put_documents(
                [
                    ManagedDocument(
                        document_name=document_title,
                        content=md_content_safe,
                        summary=summary_safe,
                        q_and_a=q_and_a_safe,
                        mindmap=mind_map_safe,
                        bullet_points=bullet_points_safe,
                    )
                ]
            )
        except Exception as db_e:
            logger.error(f"[run_workflow] Ошибка при записи в БД: {db_e}")
        logger.info("[run_workflow] Документ сохранён в менеджере документов")
        # Return safe, normalized string values to avoid None/null in UI
        return md_content_safe, summary_safe, q_and_a_safe, bullet_points_safe, mind_map_safe
    except Exception as e:
        logger.error(f"[run_workflow] Ошибка: {e}")
        return None, None, None, None, None  # обработка других ошибок, если возникнут
    finally:
        try:
            os.remove(temp_path)
            logger.info(f"[run_workflow] Временный файл удалён: {temp_path}")
        except OSError:
            await asyncio.sleep(0.1)
            try:
                os.remove(temp_path)
                logger.info(f"[run_workflow] Временный файл удалён после ожидания: {temp_path}")
            except OSError:
                logger.warning(f"[run_workflow] Не удалось удалить временный файл: {temp_path}")


def sync_run_workflow(file: io.BytesIO, document_title: str):

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("notebookllama")
    try:
        logger.info("[sync_run_workflow] Запуск обработки файла через run_workflow...")
        # Всегда создаём новый event loop для потоков/Streamlit
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Allow the workflow to be bounded by a configurable timeout to avoid UI hangs
        try:
            work_timeout = int(os.getenv('WORKFLOW_TIMEOUT', '600'))
        except Exception:
            work_timeout = 600
        try:
            result = loop.run_until_complete(asyncio.wait_for(run_workflow(file, document_title), timeout=work_timeout))
        except asyncio.TimeoutError:
            logger.error(f"[sync_run_workflow] Workflow timed out after {work_timeout} seconds")
            return None, None, None, None, None
        except Exception as e:
            logger.error(f"[sync_run_workflow] Ошибка во время выполнения workflow: {e}")
            return None, None, None, None, None
        logger.info("[sync_run_workflow] run_workflow завершён")
        return result
    except Exception as e:
        logger.error(f"[sync_run_workflow] Ошибка: {e}")
        return None, None, None, None, None






# Display the network
st.set_page_config(
    page_title="NotebookLlaMa - Home",
    page_icon="🏠",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/run-llama/notebooklm-clone/discussions/categories/general",
        "Report a bug": "https://github.com/run-llama/notebooklm-clone/issues/",
        "About": "An OSS alternative to NotebookLM that runs with the power of a flully Llama!",
    },
)
# st.sidebar.header("Home🏠")
# st.sidebar.info("To switch to the Document Chat, select it from above!🔺")
# HF model selector for fallback
if "hf_fallback_model" not in st.session_state:
    st.session_state.hf_fallback_model = os.getenv('HF_FALLBACK_MODEL', 'gpt2')

# Ollama model selector - get available models from Ollama
st.sidebar.markdown("### Ollama Model Settings")
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')

def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        import urllib.request, json
        tags_url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(tags_url)
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.getcode() == 200:
                data = json.loads(resp.read().decode('utf-8'))
                models = [model.get('name', '') for model in data.get('models', [])]
                return [m for m in models if m]  # Filter out empty strings
    except Exception:
        pass
    return ['mistral']  # Default fallback

ollama_models = get_ollama_models()
if ollama_models:
    selected_ollama_model = st.sidebar.selectbox(
        "Ollama Model (for document processing)",
        options=ollama_models,
        index=0 if 'mistral' in ollama_models else 0,
        help="Select which Ollama model to use for processing documents. This model will be used for summaries, mind maps, and Q&A generation."
    )
    st.session_state.ollama_model = selected_ollama_model
    os.environ['OLLAMA_MODEL'] = str(st.session_state.ollama_model)
    st.sidebar.caption(f"Using Ollama model: {st.session_state.ollama_model}")
else:
    st.sidebar.warning("Could not fetch Ollama models. Using default: mistral")
    st.session_state.ollama_model = 'mistral'
    os.environ['OLLAMA_MODEL'] = 'mistral'

st.sidebar.markdown("---")
st.sidebar.markdown("### HuggingFace Fallback Settings")
# provide a quick dropdown of common small models and a text input for custom
COMMON_HF_MODELS = ["gpt2", "distilgpt2", "facebook/opt-125m", "tiiuae/falcon-7b-instruct" ]
sel = st.sidebar.selectbox("HF model (quick select)", options=["(custom)"] + COMMON_HF_MODELS, index=0)
if sel and sel != "(custom)":
    st.session_state.hf_fallback_model = sel
st.session_state.hf_fallback_model = st.sidebar.text_input("HF fallback model (custom)", value=st.session_state.hf_fallback_model)
st.sidebar.caption("Choose a small CPU-friendly model for fast local fallback (e.g. distilgpt2 or gpt2).")
try:
    if st.session_state.hf_fallback_model:
        os.environ['HF_FALLBACK_MODEL'] = str(st.session_state.hf_fallback_model)
except Exception:
    pass

# Model load/save controls
if 'hf_load_status' not in st.session_state:
    st.session_state.hf_load_status = ''
def _background_load_model(model_name: str):
    try:
        from transformers import pipeline
        # try a tiny generation to trigger download and cache
        p = pipeline('text-generation', model=model_name)
        _ = p("Hello", max_new_tokens=8)
        st.session_state.hf_load_status = f"Loaded: {model_name}"
    except Exception as e:
        st.session_state.hf_load_status = f"Load failed: {e}"

col_a, col_b = st.sidebar.columns([2,1])
with col_a:
    if st.sidebar.button("Load model"):
        st.session_state.hf_load_status = f"Loading {st.session_state.hf_fallback_model}..."
        t = threading.Thread(target=_background_load_model, args=(st.session_state.hf_fallback_model,), daemon=True)
        t.start()
with col_b:
    if st.sidebar.button("Save model"):
        # persist to .env (append or update)
        try:
            env_path = Path(__file__).resolve().parents[2] / '.env'
            kv = f"HF_FALLBACK_MODEL={st.session_state.hf_fallback_model}\n"
            # write or update simple key
            if env_path.exists():
                # Read with UTF-8, handling BOM if present
                try:
                    text = env_path.read_text(encoding='utf-8-sig')  # utf-8-sig handles BOM
                except UnicodeDecodeError:
                    # Fallback: try reading as bytes and decode with error handling
                    text = env_path.read_bytes().decode('utf-8', errors='replace')
                if 'HF_FALLBACK_MODEL=' in text:
                    import re
                    text = re.sub(r"HF_FALLBACK_MODEL=.*\n?", kv, text)
                else:
                    text += '\n' + kv
                env_path.write_text(text, encoding='utf-8')
            else:
                env_path.write_text(kv, encoding='utf-8')
            st.session_state.hf_load_status = f"Saved {st.session_state.hf_fallback_model} to .env"
        except Exception as e:
            st.session_state.hf_load_status = f"Save failed: {e}"
if st.session_state.hf_load_status:
    st.sidebar.write(st.session_state.hf_load_status)
try:
    if st.session_state.hf_fallback_model:
        os.environ['HF_FALLBACK_MODEL'] = str(st.session_state.hf_fallback_model)
except Exception:
    pass

# HF generation params
if 'hf_gen_temperature' not in st.session_state:
    try:
        st.session_state.hf_gen_temperature = float(os.getenv('HF_GEN_TEMPERATURE', '0.0'))
    except Exception:
        st.session_state.hf_gen_temperature = 0.0
st.session_state.hf_gen_temperature = st.sidebar.slider('HF temperature', min_value=0.0, max_value=1.0, value=float(st.session_state.hf_gen_temperature), step=0.01)
try:
    os.environ['HF_GEN_TEMPERATURE'] = str(float(st.session_state.hf_gen_temperature))
except Exception:
    pass

if 'hf_gen_max_tokens' not in st.session_state:
    try:
        st.session_state.hf_gen_max_tokens = int(os.getenv('HF_GEN_MAX_TOKENS', '256'))
    except Exception:
        st.session_state.hf_gen_max_tokens = 256
st.session_state.hf_gen_max_tokens = st.sidebar.number_input('HF max tokens', min_value=16, max_value=2048, value=int(st.session_state.hf_gen_max_tokens))
try:
    os.environ['HF_GEN_MAX_TOKENS'] = str(int(st.session_state.hf_gen_max_tokens))
except Exception:
    pass

# additional generation params
if 'hf_gen_top_k' not in st.session_state:
    try:
        st.session_state.hf_gen_top_k = int(os.getenv('HF_GEN_TOP_K', '50'))
    except Exception:
        st.session_state.hf_gen_top_k = 50
st.session_state.hf_gen_top_k = st.sidebar.number_input('HF top_k', min_value=0, max_value=200, value=int(st.session_state.hf_gen_top_k))
try:
    os.environ['HF_GEN_TOP_K'] = str(int(st.session_state.hf_gen_top_k))
except Exception:
    pass

if 'hf_gen_top_p' not in st.session_state:
    try:
        st.session_state.hf_gen_top_p = float(os.getenv('HF_GEN_TOP_P', '0.95'))
    except Exception:
        st.session_state.hf_gen_top_p = 0.95
st.session_state.hf_gen_top_p = st.sidebar.slider('HF top_p', min_value=0.0, max_value=1.0, value=float(st.session_state.hf_gen_top_p), step=0.01)
try:
    os.environ['HF_GEN_TOP_P'] = str(float(st.session_state.hf_gen_top_p))
except Exception:
    pass

if 'hf_gen_do_sample' not in st.session_state:
    try:
        st.session_state.hf_gen_do_sample = os.getenv('HF_GEN_DO_SAMPLE', 'False').lower() in ('1','true','yes')
    except Exception:
        st.session_state.hf_gen_do_sample = False
st.session_state.hf_gen_do_sample = st.sidebar.checkbox('HF do_sample', value=st.session_state.hf_gen_do_sample)
try:
    os.environ['HF_GEN_DO_SAMPLE'] = '1' if st.session_state.hf_gen_do_sample else '0'
except Exception:
    pass
st.markdown("---")
st.markdown("## NotebookLlaMa - Home🦙")

# Helper functions to pretty-print long LLM outputs
def _clean_summary_text(s: str) -> str:
    if not s:
        return ""
    try:
        import re
        # replace repeated non-breaking markers
        s = s.replace('\uf0ff', '- ')
        # common bullet-like markers used in some PDFs
        s = s.replace('', '\n\n- ')
        # collapse many spaces
        s = re.sub(r"[ ]{2,}", ' ', s)
        # ensure paragraphs between sentences when punctuation followed by capital Cyrillic letter
        s = re.sub(r'([.!?])\s+([А-ЯЁ])', r'\1\n\n\2', s)
        # ensure newlines around headings like "ЭТАП" or all-caps short words
        s = re.sub(r'\n?\s*(ЭТАП\b)', r'\n\n\1', s)
        return s.strip()
    except Exception:
        return s


def _format_bullet_points(bp) -> str:
    # bp may be list or string
    try:
        if not bp:
            return ''
        if isinstance(bp, list):
            return '\n'.join([f'- {x}' for x in bp])
        txt = str(bp)
        # if the string already contains explicit bullets, normalize them
        txt = txt.replace('', '\n- ')
        # try to split by lines or sentences and create bullets
        import re
        lines = [l.strip() for l in re.split(r'\n{1,}|\r|\.|\n- ', txt) if l.strip()]
        # prefer lines that are not too long
        bullets = []
        for l in lines:
            # avoid repeating the same long header
            if len(l) > 400:
                # truncate long items
                bullets.append('- ' + l[:300].strip() + '...')
            else:
                bullets.append('- ' + l)
        return '\n'.join(bullets)
    except Exception:
        return str(bp)


def _try_parse_mind_map(mm_raw):
    # if mind_map is a string representation of a dict, try to parse and normalize node labels
    try:
        import ast
        if isinstance(mm_raw, str) and mm_raw.strip().startswith('{'):
            parsed = ast.literal_eval(mm_raw)
            # truncate node labels to reasonable length
            nodes = parsed.get('nodes', [])
            for n in nodes:
                if isinstance(n, dict) and 'label' in n and isinstance(n['label'], str):
                    if len(n['label']) > 200:
                        n['label'] = n['label'][:200] + '...'
            parsed['nodes'] = nodes
            return parsed
    except Exception:
        pass
    return mm_raw

# --- Presets for quick tuning (fast-summary, balanced, detailed, mindmap) ---
PRESETS = {
    'fast-summary': {
        'hf_fallback_model': 'distilgpt2',
        'hf_gen_temperature': 0.0,
        'hf_gen_max_tokens': 128,
        'hf_gen_top_k': 50,
        'hf_gen_top_p': 0.9,
        'hf_gen_do_sample': False,
    },
    'balanced-summary': {
        'hf_fallback_model': 'gpt2',
        'hf_gen_temperature': 0.05,
        'hf_gen_max_tokens': 256,
        'hf_gen_top_k': 40,
        'hf_gen_top_p': 0.92,
        'hf_gen_do_sample': False,
    },
    'detailed-summary': {
        'hf_fallback_model': 'gpt2-medium',
        'hf_gen_temperature': 0.1,
        'hf_gen_max_tokens': 512,
        'hf_gen_top_k': 50,
        'hf_gen_top_p': 0.95,
        'hf_gen_do_sample': False,
    },
    'mindmap-fast': {
        'hf_fallback_model': 'distilgpt2',
        'hf_gen_temperature': 0.0,
        'hf_gen_max_tokens': 160,
        'hf_gen_top_k': 40,
        'hf_gen_top_p': 0.9,
        'hf_gen_do_sample': False,
    }
}

def apply_preset(preset_name: str):
    """Apply preset values into session_state and environment variables."""
    p = PRESETS.get(preset_name)
    if not p:
        return
    try:
        # update model
        st.session_state.hf_fallback_model = p.get('hf_fallback_model', st.session_state.hf_fallback_model)
        os.environ['HF_FALLBACK_MODEL'] = str(st.session_state.hf_fallback_model)
    except Exception:
        pass
    try:
        st.session_state.hf_gen_temperature = float(p.get('hf_gen_temperature', st.session_state.hf_gen_temperature))
        os.environ['HF_GEN_TEMPERATURE'] = str(st.session_state.hf_gen_temperature)
    except Exception:
        pass
    try:
        st.session_state.hf_gen_max_tokens = int(p.get('hf_gen_max_tokens', st.session_state.hf_gen_max_tokens))
        os.environ['HF_GEN_MAX_TOKENS'] = str(st.session_state.hf_gen_max_tokens)
    except Exception:
        pass
    try:
        st.session_state.hf_gen_top_k = int(p.get('hf_gen_top_k', st.session_state.hf_gen_top_k))
        os.environ['HF_GEN_TOP_K'] = str(st.session_state.hf_gen_top_k)
    except Exception:
        pass
    try:
        st.session_state.hf_gen_top_p = float(p.get('hf_gen_top_p', st.session_state.hf_gen_top_p))
        os.environ['HF_GEN_TOP_P'] = str(st.session_state.hf_gen_top_p)
    except Exception:
        pass
    try:
        st.session_state.hf_gen_do_sample = bool(p.get('hf_gen_do_sample', st.session_state.hf_gen_do_sample))
        os.environ['HF_GEN_DO_SAMPLE'] = '1' if st.session_state.hf_gen_do_sample else '0'
    except Exception:
        pass

# Sidebar preset selector
st.sidebar.markdown("---")
preset_choice = st.sidebar.selectbox("Quick presets", options=['(none)', 'fast-summary', 'balanced-summary', 'detailed-summary', 'mindmap-fast'], index=0)
colp1, colp2 = st.sidebar.columns([1,1])
with colp1:
    if st.sidebar.button("Apply preset"):
        if preset_choice and preset_choice != '(none)':
            apply_preset(preset_choice)
            st.experimental_rerun()
with colp2:
    if st.sidebar.button("Apply & Load model"):
        if preset_choice and preset_choice != '(none)':
            apply_preset(preset_choice)
            # trigger background load
            t = threading.Thread(target=_background_load_model, args=(st.session_state.hf_fallback_model,), daemon=True)
            t.start()
            st.experimental_rerun()


# Show quick status hints for local services (Ollama, DB)
try:
    try:
        # cache the Ollama check for a short period to avoid false negatives on repeated imports
        if "_ollama_check_ts" in st.session_state and "_ollama_ok" in st.session_state:
            # cache for 10 seconds
            if (time.time() - st.session_state._ollama_check_ts) < 10:
                ollama_ok = st.session_state._ollama_ok
            else:
                ollama_ok = is_ollama_available()
                st.session_state._ollama_ok = ollama_ok
                st.session_state._ollama_check_ts = time.time()
        else:
            ollama_ok = is_ollama_available()
            st.session_state._ollama_ok = ollama_ok
            st.session_state._ollama_check_ts = time.time()
    except Exception:
        ollama_ok = False
    if not ollama_ok:
        col1, col2 = st.columns([6,1])
        with col1:
            st.warning("Ollama seems unreachable at http://localhost:11434 — some features (mind map, summaries) may not work. Try starting backend containers: `docker compose up -d`.")
        with col2:
            if st.button("Re-check"):
                # force a re-check and update the cached value
                new_ok = is_ollama_available()
                st.session_state._ollama_ok = new_ok
                st.session_state._ollama_check_ts = time.time()
                if new_ok:
                    st.experimental_rerun()
    else:
        st.info("Ollama endpoint is reachable (http://localhost:11434).")
    try:
        if not getattr(document_manager, "_engine", None):
            st.warning("Database not configured — results will not be saved. Set pgql_user/pgql_psw/pgql_db or run `docker compose up -d` to start local Postgres.")
        else:
            st.info("Database engine configured — results will be persisted.")
    except Exception:
        st.info("DB status unknown (could not check DocumentManager engine).")
except Exception:
    # UI check should never fail the app; swallow any unexpected errors
    pass

# Initialize session state BEFORE creating the text input
if "workflow_results" not in st.session_state:
    st.session_state.workflow_results = None
if "document_title" not in st.session_state:
    st.session_state.document_title = randomname.get_name(
        adj=("music_theory", "geometry", "emotions"), noun=("cats", "food")
    )
# Processing guard and upload hashes to avoid duplicate work
if "_processing" not in st.session_state:
    st.session_state._processing = False
if "_uploaded_hash" not in st.session_state:
    st.session_state._uploaded_hash = None
if "_last_processed_hash" not in st.session_state:
    st.session_state._last_processed_hash = None

# Use session_state as the value and update it when changed
document_title = st.text_input(
    label="Document Title",
    value=st.session_state.document_title,
    key="document_title_input",
)

# Update session state when the input changes
if document_title != st.session_state.document_title:
    st.session_state.document_title = document_title

file_input = st.file_uploader(
    label="Upload your source PDF file!", accept_multiple_files=False
)

# HF fallback toggle (optional): if enabled, workflow will try to use a local HuggingFace model when Ollama fails
if "use_hf_fallback" not in st.session_state:
    st.session_state.use_hf_fallback = False
st.session_state.use_hf_fallback = st.sidebar.checkbox("Use local HF fallback if Ollama unavailable", value=st.session_state.use_hf_fallback)

# Hybrid / extractive defaults: allow users to prefer extractive summaries and run LLM refine in background
if "always_extractive_default" not in st.session_state:
    st.session_state.always_extractive_default = False
st.session_state.always_extractive_default = st.sidebar.checkbox("Always use extractive by default (fast)", value=st.session_state.always_extractive_default)

# Expose an option to auto-apply refined LLM result when it becomes available
if "auto_apply_llm_refine" not in st.session_state:
    st.session_state.auto_apply_llm_refine = True
st.session_state.auto_apply_llm_refine = st.sidebar.checkbox("Auto-apply LLM refine result when ready", value=st.session_state.auto_apply_llm_refine)

# Ensure the operational extractive toggle defaults to the user's preference
if "use_extractive_fallback" not in st.session_state:
    st.session_state.use_extractive_fallback = bool(st.session_state.always_extractive_default)
else:
    # if user toggles the global default, prefer that when unset explicitly
    st.session_state.use_extractive_fallback = st.session_state.use_extractive_fallback or bool(st.session_state.always_extractive_default)

# Background refine state
if "_llm_refine_running" not in st.session_state:
    st.session_state._llm_refine_running = False
if "llm_refined_result" not in st.session_state:
    st.session_state.llm_refined_result = None


if file_input is not None:
    # Save raw bytes in session state to allow re-generation without re-upload
    if isinstance(file_input, io.BytesIO):
        st.session_state._uploaded_bytes = file_input.getvalue()
    else:
        try:
            st.session_state._uploaded_bytes = file_input.read()
        except Exception:
            st.session_state._uploaded_bytes = None

    # Compute a simple content hash to detect re-uploads / prevent duplicate processing
    try:
        if getattr(st.session_state, "_uploaded_bytes", None):
            h = hashlib.sha256()
            h.update(st.session_state._uploaded_bytes)
            st.session_state._uploaded_hash = h.hexdigest()
        else:
            st.session_state._uploaded_hash = None
    except Exception:
        st.session_state._uploaded_hash = None

    # Warning if DB is not configured
    try:
        from notebookllama.documents import DocumentManager
        if not document_manager._engine:
            st.warning("DB not configured — results will not be saved. To enable DB, set pgql_user/pgql_psw/pgql_db in your environment.")
    except Exception:
        pass
    # Second button (hybrid): Extractive immediately, LLM refine in background
    colx1, colx2 = st.columns([1,1])
    with colx2:
        if st.button("Extractive + LLM refine"):
            if st.session_state._processing:
                st.info("A processing job is already running. Please wait for it to finish.")
            else:
                # run extractive path immediately (same as extractive fallback)
                st.session_state._processing = True
                try:
                    from notebookllama.processing import extractive_summary, parse_file, compact_bullets, topic_segmentation, shorten_mindmap_nodes
                    import tempfile
                    import asyncio

                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    tmp.write(st.session_state._uploaded_bytes)
                    tmp.flush()
                    tmp.close()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    text, _, _ = loop.run_until_complete(parse_file(tmp.name))
                    n_sent = 3
                    if text and len(text) > 4000:
                        n_sent = 5
                    summary_text = extractive_summary(text or '', num_sentences=n_sent)
                    bullets = compact_bullets(text or '', num_bullets=6)
                    segments = topic_segmentation(text or '', max_chars=2000)
                    mind_map_obj = {'type': 'Иерархическая', 'nodes': [{'id': 'root', 'label': 'Summary'}], 'edges': []}
                    for i, seg in enumerate(segments[:6], start=1):
                        nid = f'n{i}'
                        lab = seg.split('\n')[0][:200]
                        mind_map_obj['nodes'].append({'id': nid, 'label': lab})
                        mind_map_obj['edges'].append({'from': 'root', 'to': nid, 'type': 'contains'})
                    mind_map_obj = shorten_mindmap_nodes(mind_map_obj, max_label_len=120)
                    st.session_state.workflow_results = {
                        'md_content': text or '',
                        'summary': summary_text,
                        'q_and_a': '',
                        'bullet_points': bullets,
                        'mind_map': mind_map_obj,
                    }
                    try:
                        os.remove(tmp.name)
                    except Exception:
                        pass
                    st.session_state._last_processed_hash = st.session_state._uploaded_hash
                    st.success('Extractive summary generated (fast). Background LLM refine started.')
                except Exception as ex:
                    st.warning(f"Extractive fallback failed: {ex}")
                finally:
                    st.session_state._processing = False

                # Start background refine thread that runs full LLM workflow and writes result to a temp JSON file
                # to avoid direct writes to streamlit's session_state from background threads.
                import json, tempfile, uuid

                # mark running before starting the thread (main thread owns this flag)
                st.session_state._llm_refine_running = True

                def _background_refine_to_file(bytes_blob, title, out_path):
                    try:
                        # propagate HF fallback env into workflow
                        try:
                            if st.session_state.get('use_hf_fallback'):
                                os.environ['USE_HF_FALLBACK'] = '1'
                            else:
                                if 'USE_HF_FALLBACK' in os.environ:
                                    del os.environ['USE_HF_FALLBACK']
                        except Exception:
                            pass
                        res = sync_run_workflow(io.BytesIO(bytes_blob), title)
                        # normalize into a dict similar to workflow_results
                        try:
                            if isinstance(res, (list, tuple)):
                                md_content, summary, q_and_a, bullet_points, mind_map = res
                            elif hasattr(res, 'md_content') or hasattr(res, 'summary'):
                                evt = res
                                md_content = getattr(evt, 'md_content', '')
                                summary = getattr(evt, 'summary', '')
                                q_and_a = getattr(evt, 'q_and_a', '')
                                bullet_points = getattr(evt, 'bullet_points', None) or getattr(evt, 'highlights', [])
                                mind_map = getattr(evt, 'mind_map', None)
                            else:
                                md_content = res
                                summary = ''
                                q_and_a = ''
                                bullet_points = []
                                mind_map = None
                        except Exception:
                            md_content = res
                            summary = ''
                            q_and_a = ''
                            bullet_points = []
                            mind_map = None
                        out = {
                            'md_content': md_content,
                            'summary': summary,
                            'q_and_a': q_and_a,
                            'bullet_points': bullet_points,
                            'mind_map': mind_map,
                        }
                        # write atomically
                        tmpf = out_path + f'.{uuid.uuid4().hex}.tmp'
                        with open(tmpf, 'w', encoding='utf-8') as fh:
                            json.dump(out, fh, ensure_ascii=False)
                        os.replace(tmpf, out_path)
                    except Exception as _e:
                        try:
                            with open(out_path, 'w', encoding='utf-8') as fh:
                                json.dump({'error': str(_e)}, fh)
                        except Exception:
                            pass

                # choose a temp file path based on upload hash if available, otherwise random
                import tempfile, os
                upload_hash = st.session_state.get('_uploaded_hash') or uuid.uuid4().hex
                out_path = os.path.join(tempfile.gettempdir(), f'notebookllama_llm_refine_{upload_hash}.json')
                t = threading.Thread(target=_background_refine_to_file, args=(st.session_state._uploaded_bytes, st.session_state.document_title, out_path), daemon=True)
                t.start()
    # First button: Process Document
    # Process Document button with guards to avoid duplicate processing
    if st.button("Process Document", type="primary"):
        if st.session_state._processing:
            st.info("A processing job is already running. Please wait for it to finish.")
        else:
            # If we already processed the exact same bytes, reuse results instead of reprocessing
            if (
                st.session_state._uploaded_hash
                and st.session_state._last_processed_hash
                and st.session_state._uploaded_hash == st.session_state._last_processed_hash
                and st.session_state.workflow_results
            ):
                st.info("This file was already processed — using cached results.")
            else:
                st.session_state._processing = True
                # If extractive fallback is enabled, perform a fast local extractive summary and skip LLM
                if st.session_state.get('use_extractive_fallback'):
                    try:
                        from notebookllama.processing import extractive_summary, parse_file

                        import tempfile
                        import asyncio

                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                        tmp.write(st.session_state._uploaded_bytes)
                        tmp.flush()
                        tmp.close()
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        text, _, _ = loop.run_until_complete(parse_file(tmp.name))
                        # pick number of sentences based on text length
                        n_sent = 3
                        if text and len(text) > 4000:
                            n_sent = 5
                        summary_text = extractive_summary(text or '', num_sentences=n_sent)
                        # compact bullets and topic segmentation
                        try:
                            from notebookllama.processing import compact_bullets, topic_segmentation, shorten_mindmap_nodes
                            bullets = compact_bullets(text or '', num_bullets=6)
                            segments = topic_segmentation(text or '', max_chars=2000)
                            # create a very small mind_map from segments
                            mind_map_obj = {'type': 'Иерархическая', 'nodes': [{'id': 'root', 'label': 'Summary'}], 'edges': []}
                            for i, seg in enumerate(segments[:6], start=1):
                                nid = f'n{i}'
                                lab = seg.split('\n')[0][:200]
                                mind_map_obj['nodes'].append({'id': nid, 'label': lab})
                                mind_map_obj['edges'].append({'from': 'root', 'to': nid, 'type': 'contains'})
                            mind_map_obj = shorten_mindmap_nodes(mind_map_obj, max_label_len=120)
                        except Exception:
                            bullets = []
                            mind_map_obj = None

                        st.session_state.workflow_results = {
                            'md_content': text or '',
                            'summary': summary_text,
                            'q_and_a': '',
                            'bullet_points': bullets,
                            'mind_map': mind_map_obj,
                        }
                        try:
                            os.remove(tmp.name)
                        except Exception:
                            pass
                        st.session_state._last_processed_hash = st.session_state._uploaded_hash
                        st.session_state._processing = False
                        st.success('Extractive summary generated (fast).')
                        # refresh UI to show results
                        try:
                            if hasattr(st, 'experimental_rerun'):
                                st.experimental_rerun()
                        except Exception:
                            pass
                        # skip the LLM workflow
                    except Exception as ex:
                        st.warning(f"Extractive fallback failed, will run full workflow: {ex}")
                with st.spinner("Processing document... This may take a few minutes."):
                    try:
                        # Propagate HF fallback preference into environment for the workflow
                        try:
                            if st.session_state.get('use_hf_fallback'):
                                os.environ['USE_HF_FALLBACK'] = '1'
                            else:
                                if 'USE_HF_FALLBACK' in os.environ:
                                    del os.environ['USE_HF_FALLBACK']
                        except Exception:
                            pass
                        res = sync_run_workflow(io.BytesIO(st.session_state._uploaded_bytes), st.session_state.document_title)
                        # Normalize possible return shapes: tuple/list or NotebookOutputEvent-like
                        diagnostics = {}
                        md_content = summary = q_and_a = bullet_points = mind_map = None
                        if isinstance(res, (list, tuple)):
                            try:
                                md_content, summary, q_and_a, bullet_points, mind_map = res
                            except Exception:
                                # fallback: put raw into md_content
                                md_content = res
                                summary = summary or ""
                                q_and_a = q_and_a or ""
                                bullet_points = bullet_points or []
                                mind_map = mind_map or None
                        elif hasattr(res, 'md_content') or hasattr(res, 'summary'):
                            evt = res
                            md_content = getattr(evt, 'md_content', '')
                            summary = getattr(evt, 'summary', '')
                            q_and_a = getattr(evt, 'q_and_a', '')
                            # older name was 'highlights' or 'bullet_points'
                            bullet_points = getattr(evt, 'bullet_points', None) or getattr(evt, 'highlights', [])
                            mind_map = getattr(evt, 'mind_map', None)
                            diagnostics = {
                                'raw_preview': getattr(evt, 'raw_preview', None),
                                'fallback_raw': getattr(evt, 'fallback_raw', None),
                                'repair_raw': getattr(evt, 'repair_raw', None),
                            }
                        else:
                            # unknown shape
                            md_content = res
                            summary = summary or ""
                            q_and_a = q_and_a or ""
                            bullet_points = bullet_points or []
                            mind_map = mind_map or None

                        st.session_state.workflow_results = {
                            "md_content": md_content,
                            "summary": summary,
                            "q_and_a": q_and_a,
                            "bullet_points": bullet_points,
                            "mind_map": mind_map,
                            # diagnostics placeholders (may be added by workflow)
                            "raw_preview": diagnostics.get('raw_preview'),
                            "fallback_raw": diagnostics.get('fallback_raw'),
                            "repair_raw": diagnostics.get('repair_raw'),
                            "expand_raw": diagnostics.get('expand_raw'),
                        }
                        # mark processed hash so we can skip redundant processing
                        try:
                            st.session_state._last_processed_hash = st.session_state._uploaded_hash
                        except Exception:
                            st.session_state._last_processed_hash = None
                        st.success("Document processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                    finally:
                        st.session_state._processing = False

    # Display results if available
    if st.session_state.workflow_results:
        results = st.session_state.workflow_results

        # Отладочный вывод структуры результатов
        st.write("### Debug: workflow_results", results)

        # Hybrid refine status: check for background refine temp file and show controls
        try:
            import tempfile, os, json
            upload_hash = st.session_state.get('_uploaded_hash')
            if upload_hash:
                out_path = os.path.join(tempfile.gettempdir(), f'notebookllama_llm_refine_{upload_hash}.json')
            else:
                out_path = None

            # if a temp file exists (background thread wrote the result), load it into session_state
            if out_path and os.path.exists(out_path):
                try:
                    # Try reading with utf-8, fallback to utf-8-sig and other encodings
                    try:
                        with open(out_path, 'r', encoding='utf-8') as fh:
                            refined = json.load(fh)
                    except UnicodeDecodeError:
                        try:
                            with open(out_path, 'r', encoding='utf-8-sig') as fh:
                                refined = json.load(fh)
                        except UnicodeDecodeError:
                            # Fallback: read as bytes and decode
                            with open(out_path, 'rb') as fh:
                                content = fh.read().decode('utf-8', errors='replace')
                                refined = json.loads(content)
                    # move into session state for display and remove file
                    st.session_state.llm_refined_result = refined
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    # background refine finished
                    st.session_state._llm_refine_running = False
                except Exception:
                    pass

            # show running indicator
            if st.session_state.get('_llm_refine_running'):
                st.info("Background LLM refine is running — the UI will update when the refined result is ready.")

            # if a refined result arrived in session_state, surface it and allow apply/dismiss
            if st.session_state.get('llm_refined_result') is not None:
                refined = st.session_state.llm_refined_result
                if isinstance(refined, dict) and refined.get('error'):
                    st.error(f"Background LLM refine failed: {refined.get('error')}")
                    # clear to avoid repeated messages
                    st.session_state.llm_refined_result = None
                else:
                    # Show a small preview and apply options
                    with st.expander("Background LLM refine result ready — preview"):
                        try:
                            st.write(refined)
                        except Exception:
                            st.write(str(refined))

                    # Auto-apply if user opted in
                    if st.session_state.get('auto_apply_llm_refine'):
                        try:
                            st.session_state.workflow_results = st.session_state.llm_refined_result
                            st.session_state.llm_refined_result = None
                            st.success("LLM refined result applied automatically.")
                            try:
                                if hasattr(st, 'experimental_rerun'):
                                    st.experimental_rerun()
                            except Exception:
                                pass
                        except Exception as _e:
                            st.error(f"Could not apply refined result: {_e}")
                    else:
                        c1, c2 = st.columns([1,1])
                        with c1:
                            if st.button("Apply refined result"):
                                try:
                                    st.session_state.workflow_results = st.session_state.llm_refined_result
                                    st.session_state.llm_refined_result = None
                                    st.success("LLM refined result applied.")
                                    try:
                                        if hasattr(st, 'experimental_rerun'):
                                            st.experimental_rerun()
                                    except Exception:
                                        pass
                                except Exception as _e:
                                    st.error(f"Apply failed: {_e}")
                        with c2:
                            if st.button("Dismiss refined result"):
                                st.session_state.llm_refined_result = None
                                st.info("Refined result dismissed.")
        except Exception:
            pass

        # If the workflow used any fallback/repair (e.g., Ollama timed out or returned invalid JSON),
        # show a banner with Retry and Increase timeout controls.
        try:
            fallback_used = False
            for k in ("fallback_raw", "repair_raw", "expand_raw", "raw_preview"):
                v = results.get(k)
                if v:
                    # if any diagnostic exists, consider it a fallback/repair situation
                    fallback_used = True
                    break
            if fallback_used:
                st.warning("The LLM response needed fallback/repair (Ollama may have timed out or returned invalid data). The UI used local heuristics to produce summary/mindmap.")
                c1, c2, c3 = st.columns([1,1,2])
                with c1:
                    if st.button("Retry"):
                        # Rerun the workflow; preserve HF fallback toggle and optional increased timeout
                        if st.session_state._processing:
                            st.info("A processing job is already running. Please wait for it to finish.")
                        else:
                            if not getattr(st.session_state, '_uploaded_bytes', None):
                                st.error("No uploaded bytes found; please re-upload the file and try again.")
                            else:
                                st.session_state._processing = True
                                with st.spinner("Retrying workflow (may take a while)..."):
                                    # Allow user to temporarily override timeouts via session_state
                                    try:
                                        if getattr(st.session_state, 'temp_workflow_timeout', None):
                                            os.environ['WORKFLOW_TIMEOUT'] = str(int(st.session_state.temp_workflow_timeout))
                                    except Exception:
                                        pass
                                    # honor HF fallback toggle on retry
                                    try:
                                        if st.session_state.get('use_hf_fallback'):
                                            os.environ['USE_HF_FALLBACK'] = '1'
                                        else:
                                            if 'USE_HF_FALLBACK' in os.environ:
                                                del os.environ['USE_HF_FALLBACK']
                                    except Exception:
                                        pass
                                    try:
                                        md_content, summary, q_and_a, bullet_points, mind_map = sync_run_workflow(io.BytesIO(st.session_state._uploaded_bytes), st.session_state.document_title)
                                        st.session_state.workflow_results = {
                                            "md_content": md_content,
                                            "summary": summary,
                                            "q_and_a": q_and_a,
                                            "bullet_points": bullet_points,
                                            "mind_map": mind_map,
                                        }
                                        try:
                                            st.session_state._last_processed_hash = st.session_state._uploaded_hash
                                        except Exception:
                                            pass
                                        st.experimental_rerun()
                                    except Exception as _e:
                                        st.error(f"Retry failed: {_e}")
                                    finally:
                                        st.session_state._processing = False
                with c2:
                    # Increase timeout control
                    current_to = int(os.getenv('WORKFLOW_TIMEOUT', '600'))
                    new_to = st.number_input("Workflow timeout (seconds)", min_value=30, max_value=3600, value=st.session_state.get('temp_workflow_timeout', current_to))
                    st.session_state.temp_workflow_timeout = int(new_to)
                with c3:
                    # Show HF fallback button / info
                    if st.session_state.get('use_hf_fallback'):
                        st.info("HF fallback is enabled in the sidebar. On Retry the workflow will attempt a local HF model if Ollama is unavailable.")
                    else:
                        st.info("Enable 'Use local HF fallback' in the sidebar to attempt a local HuggingFace model when Ollama fails.")
        except Exception:
            pass

        # Show diagnostic previews if present
        if results.get("raw_preview"):
            with st.expander("Raw LLM response preview"):
                st.text(results.get("raw_preview"))
        if results.get("fallback_raw"):
            with st.expander("Fallback (mindmap) LLM preview"):
                st.text(results.get("fallback_raw"))
        if results.get("repair_raw"):
            with st.expander("Repair LLM preview"):
                st.text(results.get("repair_raw"))
        if results.get("expand_raw"):
            with st.expander("Expand summary LLM preview"):
                st.text(results.get("expand_raw"))

        # Summary
        st.markdown("## Summary")
        if results["summary"]:
            cleaned = _clean_summary_text(results["summary"])
            st.markdown(cleaned)
        else:
            st.info("No summary generated.")

        # Bullet Points
        st.markdown("## Bullet Points")
        bps = results.get("bullet_points")
        if bps:
            formatted = _format_bullet_points(bps)
            st.markdown(formatted)
        else:
            st.info("No bullet points generated.")

        # FAQ (toggled)
        with st.expander("FAQ"):
            qa = results.get("q_and_a")
            if qa:
                # if already a string, render directly; if list of pairs, format
                if isinstance(qa, str):
                    st.markdown(qa)
                elif isinstance(qa, list):
                    # if list is alternating question/answer, try to pair
                    if all(isinstance(i, str) for i in qa) and len(qa) % 2 == 0:
                        md = ""
                        for i in range(0, len(qa), 2):
                            md += f"**{qa[i]}**\n\n{qa[i+1]}\n\n"
                        st.markdown(md)
                    else:
                        # fallback: print repr
                        st.write(qa)
                else:
                    st.write(qa)
            else:
                st.info("No FAQ generated.")

        # Mind Map
        st.markdown("## Mind Map")
        mm = results.get("mind_map")
        if not mm:
            st.info("No mind map generated.")
            # Regenerate mind map button: only allow when not processing
            if st.button("Regenerate Mind Map"):
                if st.session_state._processing:
                    st.info("A processing job is already running. Please wait for it to finish.")
                else:
                    # Try regenerate using previously uploaded bytes
                    if not getattr(st.session_state, "_uploaded_bytes", None):
                        st.error("Original file bytes not found; please re-upload the file and try again.")
                    else:
                        st.session_state._processing = True
                        try:
                            with st.spinner("Generating mind map..."):
                                # honor HF fallback toggle on regenerate
                                try:
                                    if st.session_state.get('use_hf_fallback'):
                                        os.environ['USE_HF_FALLBACK'] = '1'
                                    else:
                                        if 'USE_HF_FALLBACK' in os.environ:
                                            del os.environ['USE_HF_FALLBACK']
                                except Exception:
                                    pass
                                # allow temporary workflow timeout override
                                try:
                                    if getattr(st.session_state, 'temp_workflow_timeout', None):
                                        os.environ['WORKFLOW_TIMEOUT'] = str(int(st.session_state.temp_workflow_timeout))
                                except Exception:
                                    pass
                                md_content, summary, q_and_a, bullet_points, mind_map = sync_run_workflow(
                                    io.BytesIO(st.session_state._uploaded_bytes), st.session_state.document_title
                                )
                                # update only mind_map in cached results
                                if not st.session_state.workflow_results:
                                    st.session_state.workflow_results = {}
                                st.session_state.workflow_results["mind_map"] = mind_map
                                # Update last processed hash too
                                try:
                                    st.session_state._last_processed_hash = st.session_state._uploaded_hash
                                except Exception:
                                    pass
                                # Try to programmatically rerun the Streamlit script. Some Streamlit
                                # versions do not expose experimental_rerun; provide safe fallbacks.
                                try:
                                    if hasattr(st, "experimental_rerun"):
                                        st.experimental_rerun()
                                    elif hasattr(st, "query_params"):
                                        try:
                                            qp = dict(st.query_params) if st.query_params is not None else {}
                                            qp["_refresh"] = str(int(time.time()))
                                            st.query_params = qp
                                        except Exception:
                                            st.info("Mind map regenerated. Please refresh the page to see the update.")
                                    else:
                                        st.info("Mind map regenerated. Please refresh the page to see the update.")
                                except Exception as _rerun_err:
                                    st.warning(f"Could not auto-reload UI: {_rerun_err}. Please refresh the page.")
                        except Exception as e:
                            st.error(f"Error regenerating mind map: {e}")
                        finally:
                            st.session_state._processing = False
        else:
            # If mind_map is a dict (JSON), pretty-print or render minimal HTML
            # try to parse and normalize mind_map strings
            mm_norm = _try_parse_mind_map(mm)
            if isinstance(mm_norm, dict):
                st.json(mm_norm)
            else:
                try:
                    # If string HTML, show it; otherwise show as markdown (shorten very long strings)
                    if isinstance(mm_norm, str) and mm_norm.strip().startswith("<"):
                        components.html(mm_norm, height=800, scrolling=True)
                    else:
                        mm_txt = str(mm_norm)
                        if len(mm_txt) > 10000:
                            st.markdown(mm_txt[:10000] + '\n\n... (truncated)')
                        else:
                            st.markdown(mm_txt)
                except Exception:
                    st.markdown(str(mm))

            # Add Preview and Save buttons for mindmap dicts
            try:
                if isinstance(mm, dict) and mm.get('nodes') is not None:
                    # Build a tiny HTML visualizer using vis-network (CDN)
                    def build_vis_html(mindmap_dict):
                        nodes = mindmap_dict.get('nodes', [])
                        edges = mindmap_dict.get('edges', [])
                        # normalize nodes to {id,label}
                        nodes_js = []
                        for n in nodes:
                            nid = n.get('id') if isinstance(n, dict) else n
                            label = n.get('label') if isinstance(n, dict) else str(n)
                            nodes_js.append({'id': nid, 'label': label})
                        edges_js = []
                        for e in edges:
                            if isinstance(e, dict):
                                edges_js.append({'from': e.get('from'), 'to': e.get('to'), 'label': e.get('type', '')})
                        import json
                        template = f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <script type=\"text/javascript\" src=\"https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js\"></script>
    <style type=\"text/css\">#mynetwork {{ width: 100%; height: 800px; border: 1px solid lightgray; }}</style>
  </head>
  <body>
    <div id=\"mynetwork\"></div>
    <script>
      const nodes = new vis.DataSet({json.dumps(nodes_js, ensure_ascii=False)});
      const edges = new vis.DataSet({json.dumps(edges_js, ensure_ascii=False)});
      const container = document.getElementById('mynetwork');
      const data = {{ nodes: nodes, edges: edges }};
      const options = {{ physics: {{ stabilization: true }}, edges: {{ arrows: 'to' }} }};
      const network = new vis.Network(container, data, options);
    </script>
  </body>
</html>
"""
                        return template

                    vis_html = build_vis_html(mm)
                    st.markdown("### Mindmap Preview & Export")
                    # preview
                    components.html(vis_html, height=800, scrolling=True)
                    # download button
                    try:
                        st.download_button("Save mindmap as HTML", data=vis_html, file_name=(st.session_state.document_title or 'mindmap') + '.html', mime='text/html')
                    except Exception:
                        st.info("Mindmap ready — copy/paste HTML from preview to save manually.")
            except Exception:
                pass

else:
    st.info("Please upload a PDF file to get started.")


