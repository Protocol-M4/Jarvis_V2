import requests
import json
from config import settings

class LLMEngine:
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.LLM_MODEL
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Загружаем личность из файла, указанного в настройках
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self):
        """Читает системный промпт из внешнего файла"""
        try:
            with open(settings.SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"{settings.Colors.ERROR}Ошибка загрузки промпта: {e}{settings.Colors.END}")
            return "Ты — Джарвис, полезный ИИ-помощник." # Запасной вариант

    def ask(self, user_text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text}
            ],
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS
        }

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Босс, у нас проблемы со связью: {e}"
