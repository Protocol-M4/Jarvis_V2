#!/usr/bin/env python3
"""
Скрипт для загрузки голосовых моделей Piper TTS.
"""

import os
import sys
import json
import requests
from tqdm import tqdm

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def download_file(url, destination, force=False, min_size_mb=1):
    """
    Загружает файл с указанного URL и сохраняет его в указанное место.
    Показывает прогресс загрузки.
    
    Args:
        url: URL для загрузки
        destination: Путь для сохранения файла
        force: Если True, перезаписывает существующий файл
        min_size_mb: Минимальный размер файла в МБ для проверки корректности
    """
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Проверяем, существует ли файл
    if os.path.exists(destination) and not force:
        # Проверяем размер файла
        file_size = os.path.getsize(destination)
        min_size_bytes = min_size_mb * 1024 * 1024  # Конвертируем МБ в байты
        
        if file_size >= min_size_bytes:
            print(f"Файл {destination} уже существует и имеет корректный размер ({file_size / (1024*1024):.2f} МБ). Пропускаем загрузку.")
            return
        else:
            print(f"Файл {destination} существует, но имеет подозрительно малый размер ({file_size / 1024:.2f} КБ). Перезагружаем.")
    
    # Загружаем файл
    print(f"Загрузка {url}...")
    response = requests.get(url, stream=True)
    
    # Проверяем успешность запроса
    if response.status_code != 200:
        print(f"Ошибка при загрузке {url}: {response.status_code}")
        return False
    
    # Получаем размер файла
    total_size = int(response.headers.get('content-length', 0))
    
    # Загружаем файл с отображением прогресса
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # Проверяем размер загруженного файла
    file_size = os.path.getsize(destination)
    min_size_bytes = min_size_mb * 1024 * 1024  # Конвертируем МБ в байты
    
    if file_size < min_size_bytes:
        print(f"ОШИБКА: Загруженный файл {destination} имеет размер {file_size / 1024:.2f} КБ, что меньше минимального требуемого размера {min_size_mb} МБ.")
        print(f"Возможно, URL {url} неверный или сервер вернул ошибку 404 в виде HTML-страницы.")
        
        # Удаляем поврежденный файл
        os.remove(destination)
        print(f"Поврежденный файл {destination} удален.")
        return False
    
    print(f"Файл {destination} успешно загружен. Размер: {file_size / (1024*1024):.2f} МБ")
    return True

def download_piper_models(force=False, min_size_mb=1):
    """
    Загружает модели Piper TTS для русского языка.
    
    Args:
        force: Если True, перезаписывает существующие файлы
        min_size_mb: Минимальный размер файла в МБ для проверки корректности
    """
    # Создаем директорию для моделей
    models_dir = os.path.join("models", "piper")
    os.makedirs(models_dir, exist_ok=True)
    
    # Список доступных моделей и их URL
    models = [
        {
            "name": "ru_RU-dmitri-medium",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json"
        },
        {
            "name": "ru_RU-irina-medium",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json"
        }
    ]
    
    # Загружаем модели
    success_count = 0
    total_models = len(models)
    
    for model in models:
        model_name = model["name"]
        model_url = model["url"]
        config_url = model["config_url"]
        
        model_path = os.path.join(models_dir, f"{model_name}.onnx")
        config_path = os.path.join(models_dir, f"{model_name}.onnx.json")
        
        print(f"Загрузка модели {model_name}...")
        model_success = download_file(model_url, model_path, force=force, min_size_mb=min_size_mb)
        config_success = download_file(config_url, config_path, force=force, min_size_mb=0.001)  # Конфиг может быть очень маленьким (1 КБ)
        
        if model_success and config_success:
            success_count += 1
            print(f"Модель {model_name} успешно загружена.")
        else:
            print(f"Не удалось загрузить модель {model_name}.")
    
    if success_count > 0:
        print(f"Успешно загружено {success_count} из {total_models} моделей.")
    else:
        print("ВНИМАНИЕ: Не удалось загрузить ни одной модели!")
        print("Возможные причины:")
        print("1. Неверные URL-адреса (HuggingFace мог изменить структуру репозитория)")
        print("2. Проблемы с подключением к интернету")
        print("3. Модели были удалены или перемещены")
        print("\nПопробуйте следующее:")
        print("1. Проверьте доступность моделей на сайте HuggingFace: https://huggingface.co/rhasspy/piper-voices")
        print("2. Обновите URL-адреса в скрипте, если структура изменилась")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Загрузка голосовых моделей Piper TTS")
    parser.add_argument("--force", action="store_true", help="Принудительно перезагрузить модели")
    parser.add_argument("--min-size", type=float, default=1.0, help="Минимальный размер файла модели в МБ (по умолчанию 1 МБ)")
    
    args = parser.parse_args()
    
    download_piper_models(force=args.force, min_size_mb=args.min_size)
