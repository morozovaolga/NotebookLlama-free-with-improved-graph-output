import requests, os, sys

url = "http://localhost:11434/api/tags"
print("Проверка доступности Ollama:", url)
try:
    r = requests.get(url, timeout=5)
    print("Status:", r.status_code)
    print("Body (truncated):", r.text[:800])
except Exception as e:
    print("Ошибка запроса /api/tags:", repr(e))

# Небольшой тест генерации
gen_url = "http://localhost:11434/api/generate"
payload = {"model": "mistral:latest", "prompt": "Напиши одно короткое предложение на русском.", "max_tokens": 50}
print("\nПроверка генерации:", gen_url)
try:
    r = requests.post(gen_url, json=payload, timeout=20)
    print("Status:", r.status_code)
    print("Response (truncated):", r.text[:2000])
except Exception as e:
    print("Ошибка генерации:", repr(e))