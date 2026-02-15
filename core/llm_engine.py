import requests
import json
from datetime import datetime
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
        memory_distances = []
        if self.memory:
            try:
                memory_context, memory_distances = self.memory.get_relevant_context(user_text, n_results=3)
                if memory_context:
                    logger.info(f"Retrieved {len(memory_context)} memory contexts for query")
                else:
                    # Добавляем сообщение, если база пуста или ничего не найдено
                    print(f"{settings.Colors.SYSTEM}[Память] База пуста или релевантные факты не найдены{settings.Colors.END}")
            except Exception as e:
                logger.error(f"Error retrieving memory context: {e}")
        
        # Формируем сообщения для API
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Если есть контекст из памяти, добавляем его естественным образом
        if memory_context:
            context_items = []
            
            # Выводим информацию в консоль
            print(f"{settings.Colors.SYSTEM}[Память] Найдено {len(memory_context)} релевантных фактов:{settings.Colors.END}")
            
            for i, (fact, distance) in enumerate(zip(memory_context, memory_distances), 1):
                relevance = 1 - distance
                context_items.append(f"- {fact} (релевантность: {relevance:.2%})")
                
                # Выводим каждый факт в консоль
                print(f"{settings.Colors.SYSTEM}  {i}. \"{fact[:100]}...\" (релевантность: {relevance:.2%}){settings.Colors.END}")
            
            context_text = "\n".join(context_items)
            memory_message = f"Вспоминаю из нашего прошлого общения:\n{context_text}\n\nЭто может быть полезно для ответа."
            messages.append({"role": "system", "content": memory_message})
            logger.info("Added memory context to conversation")
        
        # Добавляем текущий запрос пользователя
        messages.append({"role": "user", "content": user_text})
        
        # Определение инструмента для сохранения в память
        tools = [{
            "type": "function",
            "function": {
                "name": "save_to_memory",
                "description": "Сохраняет важную информацию в долгосрочную память Джарвиса",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Текст факта для сохранения (краткая, четкая формулировка)"
                        },
                        "importance": {
                            "type": "integer",
                            "description": "Важность факта от 1 до 5, где 1 - неважно, 5 - критически важно. Факты с важностью 1 не сохраняются.",
                            "enum": [1, 2, 3, 4, 5]
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Теги для категоризации факта (например: ['предпочтения', 'работа', 'хобби'])"
                        }
                    },
                    "required": ["text", "importance", "tags"]
                }
            }
        }]
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS,
            "tools": tools,
            "tool_choice": "auto"  # Позволяем модели решать, когда использовать инструмент
        }

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            
            # Проверяем, использовала ли модель инструмент
            if 'tool_calls' in result['choices'][0]['message']:
                tool_calls = result['choices'][0]['message']['tool_calls']
                for tool_call in tool_calls:
                    try:
                        if tool_call['function']['name'] == 'save_to_memory':
                            # Парсим аргументы
                            args = json.loads(tool_call['function']['arguments'])
                            text = str(args.get('text', ''))  # Оборачиваем в str() для защиты от ошибки типа
                            importance = args.get('importance', 3)  # Значение по умолчанию 3 (средняя важность)
                            tags = args.get('tags', [])
                            
                            # Сохраняем в память, если важность > 1 и это целое число
                            if isinstance(importance, int) and importance > 1 and self.memory:
                                metadata = {
                                    'importance': importance,
                                    'tags': tags,
                                    'timestamp': datetime.now().isoformat()
                                }
                                memory_id = self.memory.add_fact(text, metadata, importance)
                                if memory_id:
                                    logger.info(f"🧠 Автоматически сохранен факт важности {importance}: {text[:50]}...")
                                    print(f"{settings.Colors.SYSTEM}[Память] Сохранен факт важности {importance}: {text[:50]}...{settings.Colors.END}")
                    except Exception as e:
                        logger.error(f"Ошибка при обработке инструмента {tool_call['function']['name']}: {e}")
                        print(f"{settings.Colors.ERROR}[Ошибка] Проблема при обработке инструмента: {e}{settings.Colors.END}")
            
            # Возвращаем основной текстовый ответ модели без повторного запроса
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Босс, у нас проблемы со связью: {e}"
