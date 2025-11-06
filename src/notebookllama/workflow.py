

import json
from typing import Annotated, List, Union
import sys
import io
import os

# Ensure stdout/stderr are UTF-8 encoded where possible (helps Windows PowerShell prints)
try:
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    else:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
except Exception:
    pass

# Quiet mode: debug prints were added during diagnosis; default to silent for normal UI runs.
# Set environment variable NL_DEBUG=1 to enable debug prints again.
DEBUG = os.getenv('NL_DEBUG', '') == '1'
if not DEBUG:
    # shadow module-local print to avoid noisy output during Streamlit runs
    def _noop(*a, **k):
        return None
    print = _noop
# Ollama request timeout (seconds)
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '30'))

# --- Локальные классы для замены workflows ---
class Event:
    pass

class StartEvent(Event):
    pass

class StopEvent(Event):
    pass

class Context:
    def write_event_to_stream(self, ev):
        pass

def step(func):
    # Декоратор для совместимости
    return func

class Workflow:
    def __init__(self, timeout=600):
        self.timeout = timeout
    async def run(self, start_event):
        # Простой вызов для совместимости
        return await self.extract_file_data(start_event, Context())


# Перемещено выше для корректного импорта

class FileInputEvent(StartEvent):
    def __init__(self, file: str):
        self.file = file

class NotebookOutputEvent(StopEvent):
    def __init__(
        self,
        mind_map: str = None,
        md_content: str = None,
        summary: str = None,
        highlights: List[str] = None,
        questions: List[str] = None,
        answers: List[str] = None,
        raw_preview: str = None,
        fallback_raw: str = None,
        repair_raw: str = None,
    ):
        self.mind_map = mind_map
        self.md_content = md_content
        self.summary = summary
        self.highlights = highlights or []
        self.questions = questions or []
        self.answers = answers or []
        # diagnostics: short previews of raw LLM outputs to ease debugging
        self.raw_preview = raw_preview
        self.fallback_raw = fallback_raw
        self.repair_raw = repair_raw
        self.expand_raw = None


class NotebookLMWorkflow(Workflow):
    @step
    async def extract_file_data(
        self,
        ev: FileInputEvent,
        ctx: Context,
    ) -> NotebookOutputEvent:
        ctx.write_event_to_stream(ev=ev)
        from notebookllama.processing import parse_file

        text, images, tables = await parse_file(ev.file)
        if text is None:
            ev_out = NotebookOutputEvent()
            ev_out.mind_map = "Unprocessable file, sorry😭"
            ev_out.md_content = ""
            ev_out.summary = ""
            ev_out.highlights = []
            ev_out.questions = []
            ev_out.answers = []
            return ev_out

        outer_error = None
        try:
            import requests
            import re

            # Универсальный подробный промт для Ollama
            ollama_prompt = """
Ты — эксперт по анализу и структурированию информации. Твоя задача — по предоставленному тексту выполнить следующие действия:

1. Сделать краткое саммари (резюме) текста на русском языке.
2. Выделить основные тезисы текста по пунктам.
3. Составить список вопросов по содержанию текста и дать на них развернутые ответы.
4. Построить майнд-карту (mind map) по связям между тезисами. Опиши структуру майнд-карты в формате JSON, где:
   - "nodes" — список узлов (тезисов),
   - "edges" — связи между узлами (например, "причина-следствие", "дополнение", "пример", "контраст").
   - Если возможно, укажи типы майнд-карт: Иерархическая, Ассоциативная, Хронологическая, Причинно-следственная.
   - Приведи структуру майнд-карты для наиболее подходящего типа, исходя из содержания текста.

5. Сформируй результат в следующем формате (JSON):
{
  "summary": "...",
  "bullet_points": ["...", "...", "..."],
  "questions": ["...", "..."],
  "answers": ["...", "..."],
  "mindmap": {
    "type": "Иерархическая/Ассоциативная/Хронологическая/Причинно-следственная",
    "nodes": [...],
    "edges": [...]
  }
}

6. Все ответы, формулировки и структура — только на русском языке.
7. Если текст слишком короткий или пустой — сообщи об этом в каждом разделе.
8. Ответ только в формате JSON, без пояснений и текста до/после!
9. Результат должен быть готов для сохранения в базу данных.

Текст для анализа:
"""
            ollama_prompt += text[:10000]  # ограничение на длину
            # Get Ollama model from environment variable (set by UI) or use default
            ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": ollama_model, "prompt": ollama_prompt},
                    stream=True,
                    timeout=OLLAMA_TIMEOUT,
                )
            except Exception as _e:
                # fail fast — record and continue to fallbacks
                try:
                    print('[DEBUG] Ollama initial request failed:', _e)
                except Exception:
                    pass
                response = None
            # If Ollama failed and user enabled HF fallback, try a local transformers generation
            hf_raw_override = None
            try:
                if (response is None or (hasattr(response, 'status_code') and response.status_code >= 400)) and os.getenv('USE_HF_FALLBACK', '') == '1':
                    try:
                        # lightweight attempt to use HuggingFace transformers locally
                        from transformers import pipeline
                        # use model name from env (or default)
                        hf_model = os.getenv('HF_FALLBACK_MODEL', 'gpt2')
                        gen = pipeline('text-generation', model=hf_model)
                        # generation params from environment (set by UI)
                        try:
                            hf_temp = float(os.getenv('HF_GEN_TEMPERATURE', '0.0'))
                        except Exception:
                            hf_temp = 0.0
                        try:
                            hf_max_tokens = int(os.getenv('HF_GEN_MAX_TOKENS', '512'))
                        except Exception:
                            hf_max_tokens = 512
                        try:
                            hf_top_k = int(os.getenv('HF_GEN_TOP_K', '50'))
                        except Exception:
                            hf_top_k = 50
                        try:
                            hf_top_p = float(os.getenv('HF_GEN_TOP_P', '0.95'))
                        except Exception:
                            hf_top_p = 0.95
                        try:
                            hf_do_sample = os.getenv('HF_GEN_DO_SAMPLE', '0') in ('1', 'true', 'True')
                        except Exception:
                            hf_do_sample = False
                        # Create a strict JSON prompt so HF models attempt to return the required shape directly
                        hf_json_prompt = (
                            "Ты — модель, которая должна вернуть ОДИН JSON-ОБЪЕКТ на русском языке с полями: summary (строка), "
                            "bullet_points (список строк), questions (список строк), answers (список строк), mindmap (объект с type/nodes/edges). "
                            "Отвечай ТОЛЬКО JSON-ом, без пояснений. Текст для анализа:\n" + text[:8000]
                        )
                        # generate; we prefer a single deterministic-ish output via sampling disabled
                        out = gen(
                            hf_json_prompt[:2000],
                            max_new_tokens=hf_max_tokens,
                            do_sample=hf_do_sample,
                            temperature=hf_temp,
                            top_k=hf_top_k,
                            top_p=hf_top_p,
                        )
                        # transformers text-generation outputs vary; try to find generated_text
                        gen_text = None
                        if isinstance(out, list) and len(out) > 0:
                            candidate = out[0]
                            if isinstance(candidate, dict):
                                gen_text = candidate.get('generated_text') or candidate.get('text') or str(candidate)
                            else:
                                gen_text = str(candidate)
                        elif isinstance(out, str):
                            gen_text = out
                        if gen_text:
                            hf_raw_override = gen_text
                    except Exception as _hf_e:
                        try:
                            print('[DEBUG] HF fallback failed or transformers not available:', _hf_e)
                        except Exception:
                            pass
                        hf_raw_override = None
            except Exception:
                hf_raw_override = None
            def safe_decode(b: bytes) -> str:
                # Try utf-8 first, then latin-1, finally replace errors
                try:
                    return b.decode("utf-8")
                except Exception:
                    try:
                        return b.decode("latin-1")
                    except Exception:
                        try:
                            return b.decode("utf-8", errors="replace")
                        except Exception:
                            return ""

            # collect raw streaming chunks; keep full raw for diagnostics
            raw_chunks = []
            result_raw = ""
            try:
                if hf_raw_override:
                    # if HF provided a textual override, use it directly
                    result_raw = hf_raw_override
                else:
                    if response is None:
                        # nothing to iterate
                        raw_chunks = []
                    else:
                        for line in response.iter_lines():
                            if not line:
                                continue
                            # line may be bytes; try safe decoding and JSON parse
                            dec = safe_decode(line)
                            raw_chunks.append(dec)
                            try:
                                data = json.loads(dec)
                                # Ollama streams JSON objects per line; 'response' may be key
                                # append textual part if present
                                if isinstance(data, dict):
                                    resp_piece = data.get("response", "")
                                    if resp_piece:
                                        raw_chunks.append(resp_piece)
                            except Exception:
                                # not a JSON object per-line, continue
                                pass
                # join collected chunks; if empty, try non-stream fallback
                result_raw = "".join(raw_chunks)
                if not result_raw:
                    try:
                        result_raw = response.text
                    except Exception:
                        result_raw = ""
            except Exception:
                # if streaming iteration itself failed, try reading full body
                try:
                    result_raw = response.text if hasattr(response, 'text') else ""
                except Exception:
                    result_raw = ""

            # Print repr to reveal hidden/control characters and exact bytes -> helps diagnose spacing in terminal
            # helper: show hex/byte preview of a string
            def _hex_preview(s, n=64):
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

            try:
                print("[DEBUG] Ollama raw response (repr):", repr(result_raw)[:2000])
                print("[DEBUG] Ollama raw bytes (hex first 80):", _hex_preview(result_raw, 80))
            except Exception:
                print("[DEBUG] Ollama raw response (repr): [unprintable]")
            # Надёжный парсер JSON: ищем все кандидаты (нежадный режим)
            # Helper: robust JSON candidate extraction. First try non-greedy regex,
            # then a balanced-brace scanner to reassemble JSON fragments split across chunks.
            def extract_json_candidates(s: str):
                if not s:
                    return []
                # remove common markdown code fences to expose inner JSON
                s_clean = re.sub(r"```\s*json\s*([\s\S]*?)```", lambda m: m.group(1), s, flags=re.IGNORECASE)
                s_clean = re.sub(r"```([\s\S]*?)```", lambda m: m.group(1), s_clean)
                s = s_clean
                candidates = re.findall(r"\{[\s\S]*?\}", s)
                if candidates:
                    return candidates
                # Balanced-brace scanner
                out = []
                depth = 0
                start = None
                for i, ch in enumerate(s):
                    if ch == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif ch == '}':
                        if depth > 0:
                            depth -= 1
                            if depth == 0 and start is not None:
                                out.append(s[start:i+1])
                                start = None
                # If nothing found, as a last resort try a broad greedy match
                if not out:
                    m = re.search(r"\{[\s\S]*\}", s)
                    if m:
                        out = [m.group(0)]
                return out

            json_candidates = extract_json_candidates(result_raw)
            # If no candidates and result_raw empty, try a non-stream POST to capture full response
            if not json_candidates and not result_raw:
                try:
                    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
                    resp2 = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": ollama_model, "prompt": ollama_prompt},
                        timeout=OLLAMA_TIMEOUT,
                    )
                    result_raw = resp2.text or result_raw
                    json_candidates = extract_json_candidates(result_raw)
                except Exception as _e:
                    try:
                        print('[DEBUG] Ollama non-stream fallback failed:', _e)
                    except Exception:
                        pass
                    result_raw = result_raw
            summary = ""
            highlights = []
            questions = []
            answers = []
            mindmap = None

            def looks_like_result(obj: dict) -> bool:
                # consider it valid if it has at least a summary or mindmap or bullet_points
                if not isinstance(obj, dict):
                    return False
                if obj.get("summary"):
                    return True
                if obj.get("mindmap"):
                    return True
                if obj.get("bullet_points"):
                    return True
                return False

            def score_candidate(obj: dict) -> int:
                # Simple scoring: prefer candidates with longer summary, bullets, q/a and valid mindmap
                score = 0
                try:
                    if not isinstance(obj, dict):
                        return -100
                    s = obj.get('summary', '') or ''
                    if isinstance(s, str) and len(s.strip()) >= 120:
                        score += 40
                    elif isinstance(s, str) and len(s.strip()) >= 40:
                        score += 10
                    bp = obj.get('bullet_points') or obj.get('bulletPoints') or []
                    if isinstance(bp, list):
                        score += min(20, 5 * len(bp))
                    qs = obj.get('questions') or []
                    ans = obj.get('answers') or []
                    if isinstance(qs, list):
                        score += min(10, 2 * len(qs))
                    if isinstance(ans, list):
                        score += min(10, 2 * len(ans))
                    mm = obj.get('mindmap') or obj.get('mind_map') or obj.get('mindMap')
                    if isinstance(mm, dict):
                        nodes = mm.get('nodes') or []
                        edges = mm.get('edges') or []
                        if isinstance(nodes, list) and isinstance(edges, list):
                            score += 30
                except Exception:
                    return 0
                return score

            def pick_best_json_candidate(candidates: list) -> Union[dict, None]:
                best = None
                best_score = -10**9
                for cand in candidates:
                    try:
                        obj = json.loads(cand)
                    except Exception:
                        continue
                    sc = score_candidate(obj)
                    # small tie-breaker: longer JSON string
                    sc += min(5, len(cand) // 200)
                    if sc > best_score:
                        best_score = sc
                        best = obj
                return best

            def extract_array_by_key(text: str, key: str):
                """Best-effort extraction of an array of strings for a given key from noisy text."""
                if not text:
                    return []
                try:
                    # look for "key": [ ... ] block
                    pattern = re.compile(r'"' + re.escape(key) + r'"\s*:\s*\[([\s\S]*?)\]', re.IGNORECASE)
                    m = pattern.search(text)
                    items = []
                    if m:
                        inner = m.group(1)
                        # find quoted strings inside
                        items = re.findall(r'"([^\"]+)"', inner)
                        if items:
                            return [it.strip() for it in items]
                        # fallback: split by commas/newlines and strip
                        parts = re.split(r',|\n', inner)
                        for p in parts:
                            p = p.strip().strip('"').strip()
                            if p:
                                items.append(p)
                        return items
                except Exception:
                    return []
                return []

            # choose the best candidate from initial outputs
            best_initial = pick_best_json_candidate(json_candidates)
            if best_initial:
                summary = best_initial.get("summary", "")
                highlights = best_initial.get("bullet_points", []) or best_initial.get("bulletPoints", []) or []
                questions = best_initial.get("questions", []) or []
                answers = best_initial.get("answers", []) or []
                mindmap = best_initial.get("mindmap", best_initial.get("mind_map", None))
            # If JSON existed but some fields are missing or summary looks poor, ask a strict JSON-repair fallback
            def looks_like_poor_summary(s: str) -> bool:
                if not s:
                    return True
                s_str = str(s).strip()
                # too short or seems to repeat title-like pattern
                if len(s_str) < 40:
                    return True
                # repeated words or truncated endings heuristic
                if s_str.endswith('...') or s_str.count('\n') > 4:
                    return True
                return False

            need_repair = False
            if not summary or looks_like_poor_summary(summary) or not highlights or not questions or not answers:
                need_repair = True

            repair_raw = None
            if need_repair:
                try:
                    try:
                        print('[DEBUG] Performing JSON-repair fallback to fill missing fields...')
                    except Exception:
                        pass
                    partial = {
                        "summary": summary or "",
                        "bullet_points": highlights or [],
                        "questions": questions or [],
                        "answers": answers or [],
                        "mindmap": mindmap or {}
                    }
                    repair_prompt = (
                        "Ты — корректировщик JSON-результатов. У тебя есть исходный текст и НЕКОТОРЫЙ ЧАСТИЧНЫЙ JSON-ОТВЕТ. "
                        "Твоя задача — вернуть ЕДИНСТВЕННЫЙ JSON-ОБЪЕКТ на русском языке, в котором заполнены поля: summary (краткое, 3-4 предложения), "
                        "bullet_points (список ключевых тезисов до 8 пунктов), questions (до 5 вопросов) и answers (соответствующие ответы). "
                        "Поле mindmap оставь как есть или сгенерируй пустой объект {} если нельзя. ОТВЕЧАЙ ТОЛЬКО JSON, НИЧЕГО БОЛЕЕ.\n\n"
                    )
                    repair_payload = repair_prompt + "PARTIAL_JSON=" + json.dumps(partial, ensure_ascii=False) + "\n\nTEXT=" + (text[:8000] if text else "")
                    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
                    try:
                        fb = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": ollama_model, "prompt": repair_payload},
                            stream=True,
                            timeout=OLLAMA_TIMEOUT,
                        )
                    except Exception as _e:
                        try:
                            print('[DEBUG] repair request failed:', _e)
                        except Exception:
                            pass
                        fb = None
                    # collect repair raw stream similarly to initial response
                    repair_chunks = []
                    for line in fb.iter_lines():
                        if not line:
                            continue
                        repair_chunks.append(safe_decode(line))
                    repair_raw = "".join(repair_chunks) or (fb.text if hasattr(fb, 'text') else "")
                    # If streaming returned empty, try a non-streamed request as fallback
                    if not repair_raw:
                        try:
                            ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
                            fb2 = requests.post(
                                "http://localhost:11434/api/generate",
                                json={"model": ollama_model, "prompt": repair_payload},
                                timeout=OLLAMA_TIMEOUT,
                            )
                            repair_raw = fb2.text or repair_raw
                        except Exception as _e2:
                            try:
                                print('[DEBUG] non-stream repair failed:', _e2)
                            except Exception:
                                pass
                            repair_raw = f"[ERROR during non-stream repair call] {_e2}"
                    try:
                        print('[DEBUG] repair raw (repr):', repr(repair_raw)[:1000])
                        print('[DEBUG] repair raw bytes (hex first 80):', _hex_preview(repair_raw, 80))
                    except Exception:
                        print('[DEBUG] repair raw: [unprintable]')
                    fb_candidates = extract_json_candidates(repair_raw)
                    # pick best repair candidate
                    best_repair = pick_best_json_candidate(fb_candidates)
                    if best_repair:
                        summary = best_repair.get("summary", summary)
                        highlights = best_repair.get("bullet_points", highlights) or best_repair.get("bulletPoints", []) or highlights
                        questions = best_repair.get("questions", questions) or []
                        answers = best_repair.get("answers", answers) or []
                        mindmap = best_repair.get("mindmap", best_repair.get("mind_map", mindmap))
                except Exception as _e:
                    print('[DEBUG] repair fallback failed:', _e)
            # If summary still looks poor after repair, run an 'expand summary' targeted prompt
            expand_raw = None
            if looks_like_poor_summary(summary):
                try:
                    try:
                        print('[DEBUG] Expanding poor summary with targeted prompt...')
                    except Exception:
                        pass
                    expand_prompt = (
                        "Дай расширенное аналитическое резюме на русском языке. "
                        "Требуется 3-5 полных предложений, развёрнутый стиль, минимум 180 символов. "
                        "Ответ — ТОЛЬКО текст резюме (без JSON).\n\nТЕКСТ="
                    )
                    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
                    try:
                        ex = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": ollama_model, "prompt": expand_prompt + (text[:6000] if text else "")},
                            stream=True,
                            timeout=OLLAMA_TIMEOUT,
                        )
                    except Exception as _e:
                        try:
                            print('[DEBUG] expand request failed:', _e)
                        except Exception:
                            pass
                        ex = None
                    expand_chunks = []
                    for line in ex.iter_lines():
                        if not line:
                            continue
                        expand_chunks.append(safe_decode(line))
                    expand_raw = "".join(expand_chunks) or (ex.text if hasattr(ex, 'text') else "")
                    if not expand_raw:
                        try:
                            ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
                            ex2 = requests.post(
                                "http://localhost:11434/api/generate",
                                json={"model": ollama_model, "prompt": expand_prompt + (text[:6000] if text else "")},
                                timeout=OLLAMA_TIMEOUT,
                            )
                            expand_raw = ex2.text or expand_raw
                        except Exception as _e2:
                            try:
                                print('[DEBUG] non-stream expand failed:', _e2)
                            except Exception:
                                pass
                            expand_raw = f"[ERROR during non-stream expand call] {_e2}"
                    # take plain text from the stream (best-effort)
                    expanded_text = expand_raw.strip()
                    if expanded_text:
                        # If expand returned JSON (some models may do that), try to parse and extract fields
                        exp_candidates = extract_json_candidates(expanded_text)
                        if exp_candidates:
                            best_exp = pick_best_json_candidate(exp_candidates)
                            if best_exp:
                                # prefer fields from parsed JSON
                                summary = best_exp.get("summary", summary)
                                highlights = best_exp.get("bullet_points", highlights) or best_exp.get("bulletPoints", []) or highlights
                                questions = best_exp.get("questions", questions) or []
                                answers = best_exp.get("answers", answers) or []
                                mindmap = best_exp.get("mindmap", best_exp.get("mind_map", mindmap))
                        else:
                            # prefer the longest line/block as summary
                            summary = expanded_text
                except Exception as _e:
                    try:
                        print('[DEBUG] expand summary failed:', _e)
                    except Exception:
                        pass
            # Plain-text fallback: if model didn't produce JSON, try to extract simple structures
            if not summary and not highlights and not questions and not answers and not mindmap:
                # try to get a short summary from the start of text
                if text:
                    summary = text[:400].strip()
                # bullet points: lines starting with '-' or '*' or numbered lists
                lines = (text or "").splitlines()
                simple_points = [ln.strip()[1:].strip() for ln in lines if ln.strip().startswith(('-', '*'))]
                if not simple_points:
                    # numbered lists like '1.' or '1)'
                    for ln in lines:
                        s = ln.strip()
                        if re.match(r"^\d+[\.)]\s+", s):
                            simple_points.append(re.sub(r"^\d+[\.)]\s+", "", s))
                if simple_points:
                    highlights = simple_points[:10]
                # questions: lines ending with '?'
                simple_qs = [ln.strip() for ln in lines if ln.strip().endswith('?')]
                if simple_qs:
                    questions = simple_qs[:10]
                # answers remain empty in simple fallback
                if not summary:
                    summary = "Не удалось извлечь структурированный JSON — использовано простое текстовое резюме."
            try:
                # sanitize short control/unicode characters that sometimes break Windows consoles
                def _sanitize_for_print(s: str) -> str:
                    try:
                        if not isinstance(s, str):
                            return repr(s)
                        # remove BOM and common zero-width / control characters
                        for ch in ('\ufeff', '\u200b', '\u200c', '\u200d'):
                            s = s.replace(ch, '')
                        return s
                    except Exception:
                        return repr(s)

                print('[DEBUG] Summary (repr):', repr(_sanitize_for_print(summary))[:800])
                print('[DEBUG] Highlights (repr):', repr(_sanitize_for_print(str(highlights)))[:800])
                print('[DEBUG] Questions (repr):', repr(_sanitize_for_print(str(questions)))[:800])
                print('[DEBUG] Answers (repr):', repr(_sanitize_for_print(str(answers)))[:800])
                print('[DEBUG] Mindmap (repr):', repr(_sanitize_for_print(str(mindmap)))[:800])
                print('[DEBUG] Document content length:', len(text) if text else 0)
            except Exception:
                # best-effort prints; don't crash on repr failures
                try:
                    print('[DEBUG] Summary:', _sanitize_for_print(summary))
                except Exception:
                    pass
            # If mindmap not produced, attempt a focused fallback call to generate only the mindmap
            mindmap_raw = None
            if not mindmap:
                try:
                    try:
                        print('[DEBUG] Mindmap absent — выполняем fallback-вызов для генерации mindmap...')
                    except Exception:
                        pass
                    # Very prescriptive fallback prompt with multiple concrete JSON examples and strict rules
                    fallback_prompt = (
                        "Ты — машинный генератор JSON-майндкарт. ОТВЕЧАЙ ТОЛЬКО ОДНИМ JSON-ОБЪЕКТОМ, НИЧЕГО БОЛЕЕ. "
                        "Ни кода, ни текста вне JSON, ни пояснений. Если ты не можешь сгенерировать майнд-карту — верни пустой объект {}.\n"
                        "Структура JSON должна включать как минимум: \"type\" (строка), \"nodes\" (список объектов с id и label), \"edges\" (список связей с from/to/type).\n"
                        "Пример 1 (минимальный):\n"
                        "{\"type\": \"Иерархическая\", \"nodes\": [{\"id\": \"n1\", \"label\": \"Корень\"}], \"edges\": []}\n"
                        "Пример 2 (ассоциативная с двумя ветвями):\n"
                        "{\"type\": \"Ассоциативная\", \"nodes\": [{\"id\": \"n1\", \"label\": \"Центральная идея\"},{\"id\":\"n2\",\"label\":\"Подтема A\"},{\"id\":\"n3\",\"label\":\"Подтема B\"}], \"edges\": [{\"from\":\"n1\",\"to\":\"n2\",\"type\":\"дополнение\"},{\"from\":\"n1\",\"to\":\"n3\",\"type\":\"пример\"}]}\n"
                        "Пример 3 (причинно-следственная, многоуровневая):\n"
                        "{\"type\": \"Причинно-следственная\", \"nodes\": [{\"id\":\"n1\",\"label\":\"Причина\"},{\"id\":\"n2\",\"label\":\"Следствие\"},{\"id\":\"n3\",\"label\":\"Деталь\"}], \"edges\": [{\"from\":\"n1\",\"to\":\"n2\",\"type\":\"вызывает\"},{\"from\":\"n2\",\"to\":\"n3\",\"type\":\"подробность\"}]}\n"
                        "Пример 4 (многоуровневая иерархия с метаданными):\n"
                        "{\"type\": \"Иерархическая\", \"nodes\": [{\"id\": \"n1\", \"label\": \"Главная идея\", \"meta\": {\"importance\": 1}}, {\"id\": \"n2\", \"label\": \"Подтема 1\"}], \"edges\": [{\"from\": \"n1\", \"to\": \"n2\", \"type\": \"содержит\"}]}\n"
                        "Пример 5 (комплексная сеть с несколькими связями):\n"
                        "{\"type\": \"Ассоциативная\", \"nodes\": [{\"id\":\"n1\",\"label\":\"A\"},{\"id\":\"n2\",\"label\":\"B\"},{\"id\":\"n3\",\"label\":\"C\"}], \"edges\": [{\"from\":\"n1\",\"to\":\"n2\",\"type\":\"пример\"},{\"from\":\"n2\",\"to\":\"n3\",\"type\":\"дополнение\"},{\"from\":\"n1\",\"to\":\"n3\",\"type\":\"контраст\"}]}\n"
                        "Отвечай строго в этом формате — только один JSON-объект. Текст для анализа:\n"
                    )
                    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
                    try:
                        fb = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": ollama_model, "prompt": fallback_prompt + text[:8000]},
                            stream=True,
                            timeout=OLLAMA_TIMEOUT,
                        )
                    except Exception as _e:
                        try:
                            print('[DEBUG] mindmap fallback request failed:', _e)
                        except Exception:
                            pass
                        fb = None
                    mindmap_chunks = []
                    for line in fb.iter_lines():
                        if not line:
                            continue
                        mindmap_chunks.append(safe_decode(line))
                    mindmap_raw = "".join(mindmap_chunks) or (fb.text if hasattr(fb, 'text') else "")
                    if not mindmap_raw:
                        try:
                            ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
                            fb2 = requests.post(
                                "http://localhost:11434/api/generate",
                                json={"model": ollama_model, "prompt": fallback_prompt + text[:8000]},
                                timeout=OLLAMA_TIMEOUT,
                            )
                            mindmap_raw = fb2.text or mindmap_raw
                        except Exception as _e2:
                            try:
                                print('[DEBUG] non-stream mindmap failed:', _e2)
                            except Exception:
                                pass
                            mindmap_raw = f"[ERROR during non-stream mindmap call] {_e2}"
                    try:
                        print('[DEBUG] fallback raw (repr):', repr(mindmap_raw)[:1000])
                        print('[DEBUG] fallback raw bytes (hex first 80):', _hex_preview(mindmap_raw, 80))
                    except Exception:
                        print('[DEBUG] fallback raw: [unprintable]')
                    fb_candidates = extract_json_candidates(mindmap_raw)
                    def is_valid_mindmap(obj):
                        return (
                            isinstance(obj, dict)
                            and isinstance(obj.get("type", None), str)
                            and isinstance(obj.get("nodes", None), list)
                            and isinstance(obj.get("edges", None), list)
                        )

                    # pick best mindmap candidate
                    best_fb = pick_best_json_candidate(fb_candidates)
                    if best_fb and is_valid_mindmap(best_fb):
                        mindmap = best_fb
                    # If still not valid, set an explicit empty object to simplify callers
                    if mindmap is None:
                        mindmap = {}
                except Exception as _e:
                    print('[DEBUG] fallback mindmap generation failed:', _e)
        except Exception as e:
            # capture outer exception so diagnostics can report it
            try:
                outer_error = str(e)
            except Exception:
                outer_error = repr(e)
            summary = text[:200] if text else ""
            highlights = []
            questions = []
            answers = []
            mindmap = None

        # Final normalization: ensure mindmap is a dict (or empty dict) for downstream code
        try:
            if isinstance(mindmap, str):
                # try to parse stringified JSON
                mindmap_candidate = None
                try:
                    mindmap_candidate = json.loads(mindmap)
                except Exception:
                    # try to extract JSON substring
                    m = re.search(r"\{[\s\S]*\}", mindmap)
                    if m:
                        try:
                            mindmap_candidate = json.loads(m.group(0))
                        except Exception:
                            mindmap_candidate = None
                if mindmap_candidate and isinstance(mindmap_candidate, dict):
                    mindmap = mindmap_candidate
                else:
                    mindmap = {}
        except Exception:
            mindmap = {}

        # Heuristic: if bullet points are empty but we have a reasonably long summary,
        # split the summary into sentences / lines to form bullet_points and a simple mindmap.
        try:
            if (not highlights or len(highlights) == 0) and summary:
                # split by newlines first, then by sentence-ending punctuation
                candidates = []
                for part in str(summary).splitlines():
                    part = part.strip()
                    if not part:
                        continue
                    # split long lines into sentences
                    parts = re.split(r"(?<=[\.\?!])\s+", part)
                    for p in parts:
                        p = p.strip()
                        if p and len(p) >= 20:
                            candidates.append(p)
                # fallback: split by semicolon or ' - '
                if not candidates:
                    for p in re.split(r";| - | – | — ", str(summary)):
                        p = p.strip()
                        if p and len(p) >= 20:
                            candidates.append(p)
                # take up to 8 bullet points
                if candidates and (not highlights or len(highlights) == 0):
                    highlights = candidates[:8]
                # create a simple mindmap from first 6 bullets
                if mindmap in (None, {}) and highlights:
                    nodes = []
                    edges = []
                    root_id = "root"
                    nodes.append({"id": root_id, "label": "Summary"})
                    for i, h in enumerate(highlights[:6], start=1):
                        nid = f"n{i}"
                        nodes.append({"id": nid, "label": h})
                        edges.append({"from": root_id, "to": nid, "type": "содержит"})
                    mindmap = {"type": "Иерархическая", "nodes": nodes, "edges": edges}
        except Exception:
            # don't fail if heuristic fails
            pass

        # return event with diagnostics trimmed
        eo = NotebookOutputEvent()
        eo.mind_map = mindmap
        eo.md_content = text
        eo.summary = summary
        eo.highlights = highlights
        eo.questions = questions
        eo.answers = answers
        # attach possible diagnostics previews
        # populate diagnostics; if processing failed, include outer_error
        err_suffix = f" [processing error: {outer_error}]" if outer_error else ""
        try:
            eo.raw_preview = (result_raw[:2000] if 'result_raw' in locals() and result_raw else "[no raw output]") + (err_suffix if (not ('result_raw' in locals() and result_raw)) else (err_suffix if outer_error else ""))
        except Exception:
            eo.raw_preview = "[error reading raw output]" + err_suffix
        try:
            eo.fallback_raw = (mindmap_raw[:2000] if 'mindmap_raw' in locals() and mindmap_raw else "[no fallback/mindmap output]") + (err_suffix if not ('mindmap_raw' in locals() and mindmap_raw) else (err_suffix if outer_error else ""))
        except Exception:
            eo.fallback_raw = "[error reading fallback output]" + err_suffix
        try:
            eo.repair_raw = (repair_raw[:2000] if 'repair_raw' in locals() and repair_raw else "[no repair output]") + (err_suffix if not ('repair_raw' in locals() and repair_raw) else (err_suffix if outer_error else ""))
        except Exception:
            eo.repair_raw = "[error reading repair output]" + err_suffix
        try:
            eo.expand_raw = (expand_raw[:2000] if 'expand_raw' in locals() and expand_raw else "[no expand output]") + (err_suffix if not ('expand_raw' in locals() and expand_raw) else (err_suffix if outer_error else ""))
        except Exception:
            eo.expand_raw = "[error reading expand output]" + err_suffix
        return eo








# class FileInputEvent(StartEvent):
#    file: str


