#!/usr/bin/env python3
"""
Скрипт для тестирования TTS движка.
"""

import os
import sys
import time
import argparse

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tts_engine import TTSEngine
from core.logger import logger

def test_tts(text, voice_name=None):
    """
    Тестирует TTS движок с указанным текстом и голосом.
    
    Args:
        text: Текст для озвучивания
        voice_name: Имя голосовой модели (без расширения)
    """
    logger.info(f"Тестирование TTS движка с текстом: '{text}'")
    
    # Создаем экземпляр TTS движка
    if voice_name:
        tts = TTSEngine(voice_name=voice_name)
    else:
        tts = TTSEngine()
    
    # Озвучиваем текст
    logger.info("Озвучиваем текст...")
    tts.say(text)
    
    logger.info("Тестирование завершено")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирование TTS движка")
    parser.add_argument("--text", default="Привет! Это тестовое сообщение для проверки синтеза речи.", help="Текст для озвучивания")
    parser.add_argument("--voice", default=None, help="Имя голосовой модели (без расширения)")
    
    args = parser.parse_args()
    
    test_tts(args.text, args.voice)