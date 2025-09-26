

import requests

def ollama_chat(messages, model="mistral", base_url="http://localhost:11434/api/chat"):
	"""
	Отправляет сообщения в Ollama API и возвращает ответ.
	messages: список словарей вида {"role": "user"/"assistant", "content": "..."}
	model: имя модели Ollama
	base_url: адрес Ollama API
	"""
	payload = {
		"model": model,
		"messages": messages
	}
	try:
		response = requests.post(base_url, json=payload, timeout=60)
		response.raise_for_status()
		data = response.json()
		return data.get("message", {}).get("content", "")
	except Exception as e:
		print(f"Ollama API error: {e}")
		return ""
