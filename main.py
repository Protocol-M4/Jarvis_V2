import os
import sys
from config import settings
from core.logger import logger

try:
    from core.stt_engine import STTEngine
    from core.llm_engine import LLMEngine
    from core.tts_engine import TTSEngine
except ImportError as e:
    logger.error(f"Ошибка импорта модулей: {e}")
    sys.exit(1)

def main():
    logger.info("Системы запущены. Инициализация протоколов...")
    
    try:
        ears = STTEngine()
        brain = LLMEngine()
        voice = TTSEngine()
    except Exception as e:
        logger.critical(f"Критическая ошибка при старте: {e}")
        return

    logger.info("Джарвис готов к работе.")
    voice.say("Я готов к работе, Сэр.")

    while True:
        try:
            # 1. Слушаем
            ears.record_to_file()
            user_text = ears.transcribe_file()
            
            if user_text:
                logger.info(f"Пользователь: {user_text}")
                
                # Проверка на выход
                if any(word in user_text.lower() for word in ["выход", "отключись", "стоп"]):
                    logger.info("Команда завершения принята.")
                    voice.say("Отключаюсь. Доброй ночи, Сэр.")
                    break
                
                # 2. Думаем
                logger.info("Джарвис думает...")
                answer = brain.ask(user_text)
                
                # --- ЗАЩИТА ОТ ГЛЮКОВ API (типа "Ра...") ---
                if not answer or len(answer.strip()) < 3:
                    logger.warning(f"Получен некорректный ответ ({answer}), запрашиваю повтор...")
                    answer = brain.ask("Повтори еще раз, возникла системная ошибка связи.")

                # 3. Отвечаем
                logger.info(f"Джарвис: {answer}")
                voice.say(answer)
            else:
                continue

        except KeyboardInterrupt:
            logger.warning("Система остановлена пользователем.")
            break
        except Exception as e:
            logger.error(f"Ошибка в цикле: {e}", exc_info=True)

if __name__ == "__main__":
    main()
