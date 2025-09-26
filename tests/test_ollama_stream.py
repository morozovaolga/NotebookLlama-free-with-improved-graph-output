import requests, json, sys

url = "http://localhost:11434/api/generate"
payload = {"model": "mistral:latest", "prompt": "Напиши одно короткое предложение на русском.", "max_tokens": 200}

try:
    r = requests.post(url, json=payload, stream=True, timeout=300)
    print("Status:", r.status_code)
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
            print("JSON chunk:", obj)
        except Exception:
            print("Chunk:", line)
except Exception as e:
    print("Ошибка генерации (stream):", repr(e))
    sys.exit(1)