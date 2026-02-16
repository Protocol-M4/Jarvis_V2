#!/usr/bin/env python3
"""
Тестовый скрипт для генерации одного аудиофайла с использованием различных TTS библиотек.
Проверяет корректность генерации и размер файла.
"""

import os
import sys
import time
import wave
import argparse
import subprocess
from gtts import gTTS
import soundfile as sf
import numpy as np

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def test_gtts(text, output_file):
    """
    Тестирует генерацию аудио с использованием gTTS (Google Text-to-Speech).
    
    Args:
        text: Текст для озвучивания
        output_file: Путь для сохранения аудиофайла
    
    Returns:
        bool: True, если генерация успешна, иначе False
    """
    try:
        print(f"Генерация аудио с использованием gTTS: '{text}'")
        # Создаем объект gTTS
        tts = gTTS(text=text, lang='ru', slow=False)
        
        # Сохраняем аудио напрямую в файл
        tts.save(output_file)
        
        # Проверяем размер файла
        file_size = os.path.getsize(output_file)
        print(f"Файл создан: {output_file}")
        print(f"Размер файла: {file_size} байт")
        
        if file_size > 1000:  # Файл должен быть больше 1 КБ
            print("Генерация успешна! Файл имеет достаточный размер.")
            return True
        else:
            print("Ошибка: файл слишком маленький, возможно, не содержит аудио.")
            return False
    except Exception as e:
        print(f"Ошибка при генерации аудио с gTTS: {e}")
        return False

def test_pyttsx3(text, output_file):
    """
    Тестирует генерацию аудио с использованием pyttsx3.
    
    Args:
        text: Текст для озвучивания
        output_file: Путь для сохранения аудиофайла
    
    Returns:
        bool: True, если генерация успешна, иначе False
    """
    try:
        print(f"Генерация аудио с использованием pyttsx3: '{text}'")
        
        # Импортируем pyttsx3 здесь, чтобы не требовать его наличия, если не используется
        import pyttsx3
        
        # Инициализируем движок
        engine = pyttsx3.init()
        
        # Устанавливаем русский голос, если доступен
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'russian' in voice.languages or 'ru' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                print(f"Установлен русский голос: {voice.name}")
                break
        
        # Сохраняем аудио напрямую в файл
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        
        # Проверяем размер файла
        file_size = os.path.getsize(output_file)
        print(f"Файл создан: {output_file}")
        print(f"Размер файла: {file_size} байт")
        
        if file_size > 1000:  # Файл должен быть больше 1 КБ
            print("Генерация успешна! Файл имеет достаточный размер.")
            return True
        else:
            print("Ошибка: файл слишком маленький, возможно, не содержит аудио.")
            return False
    except Exception as e:
        print(f"Ошибка при генерации аудио с pyttsx3: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Тестирование генерации аудиофайла")
    parser.add_argument("--text", default="Джарвис", help="Текст для озвучивания")
    parser.add_argument("--output", default="test_audio.wav", help="Путь для сохранения аудиофайла")
    parser.add_argument("--engine", default="gtts", choices=["gtts", "pyttsx3"], help="TTS движок для использования")
    
    args = parser.parse_args()
    
    if args.engine == "gtts":
        success = test_gtts(args.text, args.output)
    elif args.engine == "pyttsx3":
        success = test_pyttsx3(args.text, args.output)
    
    if success:
        print(f"Тест успешно завершен. Проверьте файл {args.output}")
    else:
        print("Тест не пройден. Проверьте ошибки выше.")

if __name__ == "__main__":
    main()