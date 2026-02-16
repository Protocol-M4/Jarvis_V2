#!/usr/bin/env python3
"""
Тестовый скрипт для проверки полной интеграции компонентов Jarvis V2.
Проверяет работу цепочки Wake Word -> STT -> LLM -> TTS.
"""

import os
import sys
import time
import threading
import numpy as np

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings
from core.logger import logger
from core.wake_word import WakeWordDetector
from core.stt_engine import STTEngine
from core.llm_engine import LLMEngine
from core.tts_engine import TTSEngine

def test_integration():
    """Тестирует полную интеграцию компонентов"""
    print("=== Тестирование полной интеграции компонентов Jarvis V2 ===")
    
    # Инициализация компонентов
    print("Инициализация компонентов...")
    
    # Инициализация STT (Speech-to-Text)
    print(f"Загрузка модели Whisper {settings.WHISPER_MODEL_SIZE}...")
    ears = STTEngine()
    
    # Инициализация LLM (Language Model)
    print("Инициализация языковой модели...")
    brain = LLMEngine()
    
    # Инициализация TTS (Text-to-Speech)
    print("Инициализация синтеза речи...")
    voice = TTSEngine()
    
    # Инициализация детектора ключевого слова
    print("Инициализация детектора ключевого слова...")
    wake_word_detector = WakeWordDetector()
    
    print("Все компоненты инициализированы успешно")
    
    # Функция обратного вызова для детектора ключевого слова
    def on_wake_word_detected():
        print(f"Обнаружено ключевое слово '{settings.WAKE_WORDS[0]}'")
        
        # Отключаем детектор ключевого слова на время ответа
        wake_word_detector.mute(duration_seconds=3.0)
        
        # Воспроизводим звук активации
        voice.say("Слушаю вас, Никита")
        
        # Записываем новую команду
        print("Запись команды...")
        recording_success = ears.record_to_file()
        
        # Если запись не удалась (не обнаружена речь), возвращаемся в режим ожидания
        if not recording_success:
            print("Речь не обнаружена, возвращаюсь в режим ожидания")
            return
        
        # Преобразуем аудио в текст
        user_text = ears.transcribe_file()
        
        if user_text:
            print(f"Пользователь: {user_text}")
            
            # Обрабатываем запрос
            print("Обрабатываю запрос...")
            
            answer = brain.ask(user_text)
            
            # Защита от глюков API
            if not answer or len(answer.strip()) < 3:
                print("Получен некорректный ответ, запрашиваю повтор...")
                answer = brain.ask("Повтори еще раз, возникла системная ошибка связи.")
            
            # Отвечаем
            print(f"Джарвис: {answer}")
            voice.say(answer)
        else:
            print("Транскрипция не удалась или текст пустой")
    
    # Запускаем детектор ключевого слова
    print("Запуск детектора ключевого слова...")
    wake_word_detector.start_monitoring(callback=on_wake_word_detected)
    
    # Ждем 30 секунд для тестирования
    print("Ожидание 30 секунд для тестирования...")
    try:
        for i in range(30):
            print(f"Осталось {30-i} секунд...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Тестирование прервано пользователем")
    
    # Останавливаем детектор ключевого слова
    print("Остановка детектора ключевого слова...")
    wake_word_detector.stop_monitoring()
    
    print("Тестирование завершено")

if __name__ == "__main__":
    test_integration()