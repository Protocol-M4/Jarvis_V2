import os
import sys
import time
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

    session_active = False
    
    while True:
        try:
            # Если сессия не активна, ждем нажатия Enter для активации
            if not session_active:
                input(f"{settings.Colors.SYSTEM}Нажмите Enter для активации Джарвиса...{settings.Colors.END}")
                session_active = True
                logger.info("Сессия активирована пользователем")
            
            # 1. Слушаем с использованием VAD
            recording_success = ears.record_to_file()
            
            # Если запись не удалась (не обнаружена речь), продолжаем слушать
            if not recording_success:
                # Если сессия была активна, но речь не обнаружена, проверяем, нужно ли закрыть сессию
                if session_active:
                    logger.info("Речь не обнаружена, продолжаем слушать")
                continue
                
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
                
                # 4. Открываем окно ожидания для продолжения диалога
                logger.info(f"Открываю окно ожидания ({settings.SESSION_TIMEOUT}с) для продолжения диалога")
                
                # Проверяем, начал ли пользователь говорить в течение окна ожидания
                speech_detected = ears.wait_for_speech(timeout=settings.SESSION_TIMEOUT)
                
                if not speech_detected:
                    # Если пользователь не начал говорить, закрываем сессию
                    session_active = False
                    logger.info("Сессия завершена по таймауту")
                    print(f"{settings.Colors.SYSTEM}[SESSION] Сессия завершена{settings.Colors.END}")
                # Если речь обнаружена, цикл продолжится с записью
                else:
                    logger.info("Обнаружено продолжение диалога")
            else:
                logger.warning("Транскрипция не удалась или текст пустой")
                continue

        except KeyboardInterrupt:
            logger.warning("Система остановлена пользователем.")
            break
        except Exception as e:
            logger.error(f"Ошибка в цикле: {e}", exc_info=True)
            # Если произошла ошибка, сбрасываем состояние сессии
            session_active = False

if __name__ == "__main__":
    main()