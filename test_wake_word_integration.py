#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции персонализированной Wake Word в Jarvis V2.
Проверяет загрузку модели jarvis_ru_universal.joblib и корректность настроек.
"""

import os
import sys
import time
import numpy as np
import joblib
from openwakeword import Model

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings
from core.logger import logger

def test_wake_word_model():
    """Тестирует загрузку и работу модели wake word"""
    print("=== Тестирование интеграции Wake Word ===")
    
    # Проверяем наличие модели
    model_path = os.path.join("models", "openwakeword", "jarvis_ru_universal.joblib")
    if not os.path.exists(model_path):
        print(f"Ошибка: Модель не найдена по пути {model_path}")
        return False
    
    print(f"Модель найдена: {model_path}")
    
    # Проверяем настройки
    print(f"Текущий порог срабатывания (WAKE_WORD_THRESHOLD): {settings.WAKE_WORD_THRESHOLD}")
    if settings.WAKE_WORD_THRESHOLD != 0.4:
        print(f"Предупреждение: Порог срабатывания отличается от ожидаемого (0.4)")
    
    # Загружаем модель
    try:
        print("Загрузка базовой модели OpenWakeWord...")
        # Загружаем базовую модель hey_jarvis
        oww_model = Model(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx"
        )
        
        print("Модель успешно загружена")
        
        # Создаем тестовый аудио-массив (тишина)
        print("Тестирование предсказания на тишине...")
        silence = np.zeros(16000, dtype=np.float32)  # 1 секунда тишины при 16кГц
        prediction = oww_model.predict(silence)
        
        # Получаем вероятность для нашей модели
        print(f"Доступные ключи в предсказании: {list(prediction.keys())}")
        
        if "hey_jarvis" in prediction:
            score = prediction["hey_jarvis"]
            print(f"Вероятность на тишине (hey_jarvis): {score:.4f}")
        else:
            # Если модель не найдена, берем первую
            model_name = list(prediction.keys())[0]
            score = prediction[model_name]
            print(f"Вероятность на тишине ({model_name}): {score:.4f}")
        
        # Проверяем, можно ли загрузить модель напрямую через joblib
        print("\nПроверка загрузки модели через joblib...")
        try:
            model_data = joblib.load(model_path)
            print(f"Модель успешно загружена через joblib")
            print(f"Тип модели: {type(model_data)}")
            print(f"Содержимое модели: {model_data.keys() if hasattr(model_data, 'keys') else 'Не словарь'}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели через joblib: {e}")
            return False
    
    except Exception as e:
        print(f"Ошибка при тестировании модели: {e}")
        return False

if __name__ == "__main__":
    test_wake_word_model()