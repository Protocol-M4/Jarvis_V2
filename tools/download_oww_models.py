#!/usr/bin/env python3
"""
Скрипт для скачивания моделей OpenWakeWord.
Скачивает все стандартные модели, включая ONNX версии.
"""

import os
import ssl
import sys

# Обход проверки SSL для решения проблемы с LibreSSL
ssl._create_default_https_context = ssl._create_unverified_context

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_oww_models():
    """
    Скачивает все модели OpenWakeWord, включая ONNX версии.
    """
    try:
        from openwakeword.utils import download_models
        
        print("Скачивание всех стандартных моделей OpenWakeWord...")
        download_models()
        
        print("Принудительное скачивание моделей alexa...")
        download_models(model_names=['alexa'])
        
        print("Принудительное скачивание моделей hey_jarvis...")
        download_models(model_names=['hey_jarvis'])
        
        # Проверяем наличие ONNX моделей
        import pathlib
        models_dir = os.path.join(pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent, 
                                 "venv", "lib", "python3.9", "site-packages", 
                                 "openwakeword", "resources", "models")
        
        onnx_files = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
        
        if onnx_files:
            print(f"Найдены ONNX модели: {onnx_files}")
        else:
            print("ONNX модели не найдены. Проверьте директорию вручную.")
        
        return True
    except Exception as e:
        print(f"Ошибка при скачивании моделей: {e}")
        return False

if __name__ == "__main__":
    download_oww_models()