import requests
import json
from config import settings
from core.memory_manager import MemoryManager
from core.logger import logger

class LLMEngine:
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.LLM_MODEL
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Загружаем личность из файла, указанного в настройках
        self.system_prompt = self._load_system_prompt()
        
        # Инициализируем долгосрочную память
        try:
            self.memory = MemoryManager()
            logger.info("MemoryManager successfully initialized in LLMEngine")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")
            self.memory = None

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
        
        # Получаем релевантный контекст из долгосрочной памяти
        memory_context = []
        if self.memory:
            try:
                memory_context = self.memory.get_relevant_context(user_text, n_results=3)
                if memory_context:
                    logger.info(f"Retrieved {len(memory_context)} memory contexts for query")
            except Exception as e:
                logger.error(f"Error retrieving memory context: {e}")
        
        # Формируем сообщения для API
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Если есть контекст из памяти, добавляем его естественным образом
        if memory_context:
            context_text = "\n".join([f"- {fact}" for fact in memory_context])
            memory_message = f"Вспоминаю из нашего прошлого общения:\n{context_text}\n\nЭто может быть полезно для ответа."
            messages.append({"role": "system", "content": memory_message})
            logger.info("Added memory context to conversation")
        
        # Добавляем текущий запрос пользователя
        messages.append({"role": "user", "content": user_text})
        
        data = {
            "model": self.model,
            "messages": messages,
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
