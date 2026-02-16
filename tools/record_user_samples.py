#!/usr/bin/env python3
"""
Скрипт для записи голоса пользователя, произносящего "Джарвис".
Записывает 10 образцов для использования в обучении модели.
"""

import os
import sys
import time
import wave
import pyaudio
import numpy as np
import soundfile as sf
import argparse

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Параметры записи
SAMPLE_RATE = 16000  # OpenWakeWord требует 16kHz
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Моно
RECORD_SECONDS = 2  # Длительность записи каждого образца

def record_audio(output_file, seconds=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """
    Записывает аудио с микрофона и сохраняет в файл.
    
    Args:
        output_file (str): Путь для сохранения аудиофайла
        seconds (int): Длительность записи в секундах
        sample_rate (int): Частота дискретизации
    
    Returns:
        bool: True, если запись успешна, иначе False
    """
    p = pyaudio.PyAudio()
    
    # Открываем поток для записи
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print(f"Запись начнется через 3 секунды...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print(f"Запись... (говорите 'Джарвис')")
    
    frames = []
    for i in range(0, int(sample_rate / CHUNK_SIZE * seconds)):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)
    
    print("Запись завершена!")
    
    # Останавливаем и закрываем поток
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Сохраняем аудио в файл
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Аудио сохранено в {output_file}")
    return True

def record_samples(output_dir, num_samples=10):
    """
    Записывает несколько образцов голоса пользователя.
    
    Args:
        output_dir (str): Директория для сохранения образцов
        num_samples (int): Количество образцов для записи
    """
    # Создаем директорию для сохранения образцов
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Запись {num_samples} образцов произношения 'Джарвис'")
    print("Говорите четко и естественно")
    
    for i in range(num_samples):
        print(f"\nОбразец {i+1}/{num_samples}")
        output_file = os.path.join(output_dir, f"user_jarvis_{i:02d}.wav")
        record_audio(output_file)
        
        # Небольшая пауза между записями
        if i < num_samples - 1:
            print("Подготовка к следующей записи...")
            time.sleep(1)
    
    print(f"\nВсе {num_samples} образцов успешно записаны в {output_dir}")
    print("Теперь вы можете использовать эти образцы для обучения модели")

def main():
    parser = argparse.ArgumentParser(description="Запись образцов голоса пользователя")
    parser.add_argument("--output", default="data/user_samples", help="Директория для сохранения образцов")
    parser.add_argument("--samples", type=int, default=10, help="Количество образцов для записи")
    
    args = parser.parse_args()
    
    record_samples(args.output, args.samples)

if __name__ == "__main__":
    main()