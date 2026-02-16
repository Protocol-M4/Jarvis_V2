#!/usr/bin/env python3
"""
Скрипт для записи пользовательских сэмплов голоса для обучения модели Wake Word.
Записывает указанное количество аудиофайлов с голосом пользователя, произносящего ключевое слово.
"""

import os
import sys
import time
import wave
import argparse
import numpy as np
import pyaudio
from tqdm import tqdm

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def record_audio(output_dir, num_samples=15, duration=2.0, sample_rate=16000):
    """
    Записывает аудиофайлы с голосом пользователя.
    
    Args:
        output_dir (str): Директория для сохранения записанных файлов
        num_samples (int): Количество сэмплов для записи
        duration (float): Длительность каждой записи в секундах
        sample_rate (int): Частота дискретизации (16000 Гц для OpenWakeWord)
    """
    # Создаем директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Параметры аудио
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    # Инициализация PyAudio
    p = pyaudio.PyAudio()
    
    print(f"\n=== Запись {num_samples} сэмплов голоса ===")
    print(f"Каждый сэмпл будет длиться {duration} секунд")
    print("Произносите 'Джарвис' четко и естественно")
    print("=" * 50)
    
    for i in range(num_samples):
        # Имя файла для текущего сэмпла
        filename = os.path.join(output_dir, f"jarvis_user_{i:04d}.wav")
        
        # Ожидаем, пока пользователь будет готов
        input(f"\nНажмите Enter для начала записи сэмпла {i+1}/{num_samples}...")
        
        print("Подготовка к записи...")
        time.sleep(1)
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("Говорите 'Джарвис' сейчас!")
        
        # Открываем поток для записи
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Записываем аудио
        frames = []
        chunks = int(sample_rate / CHUNK_SIZE * duration)
        
        for _ in tqdm(range(chunks), desc=f"Запись сэмпла {i+1}"):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
        
        # Закрываем поток
        stream.stop_stream()
        stream.close()
        
        # Сохраняем аудио в WAV файл
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Сэмпл {i+1} сохранен в {filename}")
    
    # Закрываем PyAudio
    p.terminate()
    
    print("\n=== Запись завершена ===")
    print(f"Записано {num_samples} сэмплов в директорию {output_dir}")
    print("=" * 50)
    
    return True

def main():
    """Основная функция скрипта"""
    parser = argparse.ArgumentParser(description="Запись пользовательских сэмплов голоса")
    parser.add_argument("--output", default="data/my_samples", 
                        help="Директория для сохранения записанных файлов")
    parser.add_argument("--samples", type=int, default=15,
                        help="Количество сэмплов для записи")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Длительность каждой записи в секундах")
    parser.add_argument("--rate", type=int, default=16000,
                        help="Частота дискретизации (16000 Гц для OpenWakeWord)")
    
    args = parser.parse_args()
    
    record_audio(args.output, args.samples, args.duration, args.rate)

if __name__ == "__main__":
    main()