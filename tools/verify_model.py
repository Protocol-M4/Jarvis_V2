#!/usr/bin/env python3
"""
Скрипт для проверки точности обученной модели OpenWakeWord.
Тестирует модель на случайных примерах из датасета и в реальном времени.
"""

import os
import time
import random
import numpy as np
import sounddevice as sd
import soundfile as sf
from openwakeword import Model
from tqdm import tqdm
import argparse
import sys

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def test_on_files(model_path, dataset_dir, num_samples=20, threshold=0.6):
    """
    Тестирует модель на случайных файлах из датасета.
    
    Args:
        model_path (str): Путь к обученной модели
        dataset_dir (str): Путь к директории с датасетом
        num_samples (int): Количество случайных примеров для тестирования
        threshold (float): Порог вероятности для положительного срабатывания
    """
    # Проверяем наличие модели
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена по пути {model_path}")
        return False
    
    # Проверяем наличие директорий с данными
    positive_dir = os.path.join(dataset_dir, "positive")
    negative_dir = os.path.join(dataset_dir, "negative")
    
    if not os.path.exists(positive_dir) or not os.path.exists(negative_dir):
        print(f"Ошибка: директории с данными не найдены в {dataset_dir}")
        return False
    
    # Получаем списки файлов
    positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.wav')]
    negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.wav')]
    
    if not positive_files or not negative_files:
        print(f"Ошибка: не найдены аудиофайлы для тестирования")
        return False
    
    # Загружаем модель
    print(f"Загрузка модели {model_path}...")
    model_name = os.path.basename(model_path).split('.')[0]
    model = Model(
        wakeword_models={model_name: model_path},
        inference_framework="onnx"
    )
    
    # Выбираем случайные файлы для тестирования
    test_positive = random.sample(positive_files, min(num_samples, len(positive_files)))
    test_negative = random.sample(negative_files, min(num_samples, len(negative_files)))
    
    # Тестируем на позитивных примерах
    print("\n=== Тестирование на позитивных примерах ===")
    true_positives = 0
    
    for file in tqdm(test_positive):
        audio, sr = sf.read(file)
        
        # Нормализуем аудио
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio / np.max(np.abs(audio) + 1e-10)
        
        # Получаем предсказания
        prediction = model.predict(audio)
        score = prediction[model_name]
        
        print(f"Файл: {os.path.basename(file)}, Вероятность: {score:.4f}")
        
        if score > threshold:  # Порог для положительного срабатывания
            true_positives += 1
    
    # Тестируем на негативных примерах
    print("\n=== Тестирование на негативных примерах ===")
    true_negatives = 0
    
    for file in tqdm(test_negative):
        audio, sr = sf.read(file)
        
        # Нормализуем аудио
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio / np.max(np.abs(audio) + 1e-10)
        
        # Получаем предсказания
        prediction = model.predict(audio)
        score = prediction[model_name]
        
        print(f"Файл: {os.path.basename(file)}, Вероятность: {score:.4f}")
        
        if score <= threshold:  # Порог для отрицательного срабатывания
            true_negatives += 1
    
    # Вычисляем метрики
    precision = true_positives / len(test_positive) if len(test_positive) > 0 else 0
    specificity = true_negatives / len(test_negative) if len(test_negative) > 0 else 0
    
    print(f"\nТочность на позитивных примерах: {precision:.2%}")
    print(f"Точность на негативных примерах: {specificity:.2%}")
    print(f"Общая точность: {(true_positives + true_negatives) / (len(test_positive) + len(test_negative)):.2%}")
    
    return True

def test_live(model_path, duration=10, threshold=0.6):
    """
    Тестирует модель в реальном времени.
    
    Args:
        model_path (str): Путь к обученной модели
        duration (int): Длительность записи в секундах
        threshold (float): Порог вероятности для положительного срабатывания
    """
    # Проверяем наличие модели
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена по пути {model_path}")
        return False
    
    # Загружаем модель
    print(f"Загрузка модели {model_path}...")
    model_name = os.path.basename(model_path).split('.')[0]
    model = Model(
        wakeword_models={model_name: model_path},
        inference_framework="onnx"
    )
    
    # Параметры записи
    sample_rate = 16000
    channels = 1
    
    print(f"\n=== Тестирование в реальном времени ===")
    print(f"Запись {duration} секунд аудио. Скажите 'Джарвис'...")
    
    # Запись аудио
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()
    
    # Нормализуем аудио
    audio = audio.flatten()
    audio = audio / np.max(np.abs(audio) + 1e-10)
    
    # Получаем предсказания
    start_time = time.time()
    prediction = model.predict(audio)
    end_time = time.time()
    
    score = prediction[model_name]
    print(f"Вероятность: {score:.4f}")
    print(f"Время обработки: {(end_time - start_time)*1000:.2f} мс")
    
    # Сохраняем аудио для дальнейшего анализа
    output_file = "test_live.wav"
    sf.write(output_file, audio, sample_rate)
    print(f"Аудио сохранено в {output_file}")
    
    return True

def main(model_path, dataset_dir, live_test=True, duration=10, threshold=0.6):
    """
    Основная функция для проверки модели.
    
    Args:
        model_path (str): Путь к обученной модели
        dataset_dir (str): Путь к директории с датасетом
        live_test (bool): Проводить ли тестирование в реальном времени
        duration (int): Длительность записи в секундах для live_test
        threshold (float): Порог вероятности для положительного срабатывания
    """
    print("=== Проверка модели OpenWakeWord ===")
    
    # Тестируем на файлах из датасета
    test_on_files(model_path, dataset_dir, threshold=threshold)
    
    # Тестируем в реальном времени
    if live_test:
        test_live(model_path, duration, threshold)
    
    print("\nПроверка модели завершена")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка модели OpenWakeWord")
    parser.add_argument("--model", default="models/openwakeword/jarvis_ru.joblib", help="Путь к обученной модели")
    parser.add_argument("--dataset", default="data/wake_word_dataset", help="Директория с датасетом")
    parser.add_argument("--live", action="store_true", help="Проводить тестирование в реальном времени")
    parser.add_argument("--duration", type=int, default=10, help="Длительность записи в секундах для live_test")
    parser.add_argument("--threshold", type=float, default=0.6, help="Порог вероятности для положительного срабатывания")
    
    args = parser.parse_args()
    
    main(args.model, args.dataset, args.live, args.duration, args.threshold)