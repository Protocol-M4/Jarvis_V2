#!/usr/bin/env python3
"""
Скрипт для обучения модели OpenWakeWord на сгенерированном датасете.
Использует функцию train_custom_verifier из библиотеки openwakeword.
Оптимизирован для создания универсального русского верификатора на базе hey_jarvis.
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import argparse
import sys
from openwakeword import train_custom_verifier

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def train_model(dataset_dir, output_dir, model_name="jarvis_ru_universal", inference_framework="onnx", 
                base_model="hey_jarvis", user_samples_dir=None, user_samples_weight=75):
    """
    Обучает модель OpenWakeWord на сгенерированном датасете.
    
    Args:
        dataset_dir (str): Путь к директории с датасетом
        output_dir (str): Путь для сохранения обученной модели
        model_name (str): Имя модели
        inference_framework (str): Фреймворк для инференса ("onnx" или "tflite")
        base_model (str): Базовая модель для обучения верификатора
        user_samples_dir (str): Директория с пользовательскими сэмплами
        user_samples_weight (int): Вес пользовательских сэмплов (количество дублирований)
    """
    # Обход проверки SSL для решения проблемы с LibreSSL
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # Проверяем наличие директорий с данными
    positive_dir = os.path.join(dataset_dir, "positive")
    negative_dir = os.path.join(dataset_dir, "negative")
    
    if not os.path.exists(positive_dir) or not os.path.exists(negative_dir):
        print(f"Ошибка: директории с данными не найдены в {dataset_dir}")
        print(f"Ожидаемые пути: {positive_dir} и {negative_dir}")
        return False
    
    # Создаем директорию для сохранения модели
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем списки файлов
    positive_files = glob.glob(os.path.join(positive_dir, "*.wav"))
    negative_files = glob.glob(os.path.join(negative_dir, "*.wav"))
    
    if not positive_files or not negative_files:
        print(f"Ошибка: не найдены аудиофайлы для обучения")
        print(f"Позитивные примеры: {len(positive_files)}")
        print(f"Негативные примеры: {len(negative_files)}")
        return False
    
    # Добавляем пользовательские сэмплы, если указана директория
    user_samples = []
    if user_samples_dir and os.path.exists(user_samples_dir):
        user_samples = glob.glob(os.path.join(user_samples_dir, "*.wav"))
        if user_samples:
            print(f"Найдено {len(user_samples)} пользовательских сэмплов в {user_samples_dir}")
            
            # Дублируем пользовательские сэмплы для увеличения их веса
            weighted_user_samples = []
            for sample in user_samples:
                # Добавляем каждый пользовательский сэмпл несколько раз
                weighted_user_samples.extend([sample] * user_samples_weight)
            
            print(f"Добавляем пользовательские сэмплы с весом {user_samples_weight}")
            print(f"Общее количество пользовательских сэмплов после взвешивания: {len(weighted_user_samples)}")
            
            # Добавляем взвешенные пользовательские сэмплы к позитивным примерам
            positive_files.extend(weighted_user_samples)
    
    print(f"Итого для обучения: {len(positive_files)} позитивных и {len(negative_files)} негативных примеров")
    
    # Путь для сохранения модели
    output_path = os.path.join(output_dir, f"{model_name}.joblib")
    
    # Проверяем наличие базовых моделей и скачиваем их при необходимости
    try:
        from openwakeword.utils import download_models
        print("Проверяем наличие базовых моделей...")
        download_models()  # Скачивает как TFLite, так и ONNX модели
        print("Базовые модели готовы")
    except Exception as e:
        print(f"Предупреждение: не удалось скачать базовые модели: {e}")
        print("Продолжаем с имеющимися моделями...")
    
    # Обучаем модель
    print(f"Начинаем обучение модели {model_name} с использованием {inference_framework}...")
    print(f"Training verifier for base model: {base_model}")
    try:
        # Используем функцию train_custom_verifier из OpenWakeWord
        # Передаем списки файлов вместо директорий
        train_custom_verifier(
            positive_reference_clips=positive_files,
            negative_reference_clips=negative_files,
            output_path=output_path,
            model_name=base_model,  # Используем предобученную модель hey_jarvis как основу
            inference_framework=inference_framework  # Явно указываем использование ONNX
        )
        
        print(f"Модель успешно обучена и сохранена в {output_path}")
        return True
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели OpenWakeWord")
    parser.add_argument("--dataset", default="data/wake_word_dataset", help="Директория с датасетом")
    parser.add_argument("--output", default="models/openwakeword", help="Директория для сохранения модели")
    parser.add_argument("--name", default="jarvis_ru_universal", help="Имя модели")
    parser.add_argument("--framework", default="onnx", choices=["onnx", "tflite"], 
                        help="Фреймворк для инференса (onnx или tflite)")
    parser.add_argument("--base", default="hey_jarvis", help="Базовая модель для обучения верификатора")
    parser.add_argument("--user-samples", default="data/my_samples", 
                        help="Директория с пользовательскими сэмплами")
    parser.add_argument("--user-weight", type=int, default=75, 
                        help="Вес пользовательских сэмплов (количество дублирований)")
    
    args = parser.parse_args()
    
    train_model(args.dataset, args.output, args.name, args.framework, args.base, 
                args.user_samples, args.user_weight)
