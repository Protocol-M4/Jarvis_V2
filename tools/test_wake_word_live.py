#!/usr/bin/env python3
"""
Скрипт для тестирования модели Wake Word (Джарвис) в реальном времени.
Использует микрофон для захвата аудио и модель OpenWakeWord для обнаружения ключевого слова.
Поддерживает как базовую модель hey_jarvis, так и обученный русский верификатор jarvis_ru.
"""

# Отключаем логи ONNX
import os
os.environ['ORT_LOGGING_LEVEL'] = '3'

import sys
import time
import warnings
import numpy as np
import pyaudio
import argparse
import datetime
from openwakeword import Model

# Отключаем предупреждения для более чистого вывода
warnings.filterwarnings('ignore')

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def print_detection_message(score, model_name):
    """Выводит компактное сообщение об обнаружении ключевого слова"""
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n[SUCCESS] Модель: {model_name} | Score: {score:.2f} | Время: {current_time}")

def test_wake_word_live(threshold=0.3, use_russian_model=False):
    """
    Тестирует модель Wake Word в реальном времени с использованием микрофона.
    
    Args:
        threshold (float): Порог вероятности для срабатывания (0.0 - 1.0)
        use_russian_model (bool): Использовать ли русскую модель jarvis_ru
    """
    # Определяем, какую модель загружать
    if use_russian_model:
        print("Загрузка русской модели 'jarvis_ru'...")
        model_path = os.path.join(settings.BASE_DIR, "models", "openwakeword", "jarvis_ru.joblib")
        
        if not os.path.exists(model_path):
            print(f"Ошибка: Файл модели не найден по пути {model_path}")
            return False
        
        try:
            # Загружаем базовую модель hey_jarvis
            model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx"
            )
            
            # Пытаемся загрузить верификатор
            try:
                # Проверяем, есть ли метод для загрузки верификатора
                if hasattr(model, 'add_custom_verifier'):
                    model.add_custom_verifier(model_path)
                    print("Верификатор загружен с помощью add_custom_verifier")
                elif hasattr(model, 'load_custom_verifier'):
                    model.load_custom_verifier(model_path)
                    print("Верификатор загружен с помощью load_custom_verifier")
                else:
                    print("Методы для загрузки верификатора не найдены")
            except Exception as e:
                print(f"Ошибка при загрузке верификатора: {e}")
            
            # Используем ключ для получения предсказаний
            model_key = "hey_jarvis"
            print(f"Используем модель: {model_key}")
            
        except Exception as e:
            print(f"Ошибка при загрузке русской модели: {e}")
            return False
    else:
        print("Загрузка базовой модели 'hey_jarvis'...")
        
        try:
            # Загружаем только модель hey_jarvis без верификатора
            model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx"
            )
            
            # Выводим список активных моделей
            active_models = list(model.models.keys())
            print(f"Активные модели: {active_models}")
            
            if not active_models:
                print("Ошибка: не найдено активных моделей")
                return False
            
            # Используем ключ для получения предсказаний
            model_key = "hey_jarvis"
            print(f"Используем модель: {model_key}")
            
        except Exception as e:
            print(f"Ошибка при загрузке базовой модели: {e}")
            return False
    
    # Параметры аудио
    SAMPLE_RATE = 16000  # OpenWakeWord ожидает 16kHz
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16  # Изменено с paFloat32 на paInt16
    CHANNELS = 1  # Моно
    
    # Инициализация PyAudio
    p = pyaudio.PyAudio()
    
    # Открываем поток для записи аудио
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("\n=== Тестирование Wake Word в реальном времени ===")
    print(f"Порог срабатывания: {threshold}")
    print("Говорите 'Джарвис' для проверки. Нажмите Ctrl+C для выхода.")
    print("-" * 50)
    
    # Переменная для отслеживания времени последнего обнаружения
    last_detection_time = 0
    cooldown_period = 2.0  # Период задержки в секундах
    
    try:
        while True:
            # Читаем аудио из микрофона напрямую в формате Int16 без усиления
            audio_data = np.frombuffer(stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16)
            
            # Рассчитываем уровень громкости (RMS)
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
            
            # Получаем предсказание от модели
            prediction = model.predict(audio_data)
            
            # Выводим все ключи в предсказании для отладки (только один раз)
            if 'debug_keys_printed' not in locals():
                debug_keys_printed = True
                print(f"\nДоступные ключи в модели: {list(prediction.keys())}")
                if use_russian_model:
                    print(f"Используем модель с русским верификатором")
                else:
                    print(f"Используем чистую модель без верификатора")
                print("-" * 50)
            
            # Получаем score из модели
            if model_key in prediction:
                score = prediction[model_key]
                
                # Выводим score и уровень громкости
                print(f"\rScore: {score:.4f} | RMS: {rms:.2f}", end="")
                
                # Проверяем, прошло ли достаточно времени с последнего обнаружения
                current_time = time.time()
                time_since_last_detection = current_time - last_detection_time
                
                # Если score превышает порог и прошло достаточно времени с последнего обнаружения
                if score > threshold and time_since_last_detection > cooldown_period:
                    print_detection_message(score, model_key)
                    last_detection_time = current_time  # Обновляем время последнего обнаружения
            else:
                # Если ключ не найден, выводим доступные ключи
                available_keys = list(prediction.keys())
                print(f"\rДоступные ключи: {available_keys}, но {model_key} не найден", end="")
            
            # Небольшая задержка для снижения нагрузки на CPU
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\nЗавершение работы...")
    finally:
        # Закрываем поток и PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Тестирование завершено")

def main():
    """Основная функция скрипта"""
    parser = argparse.ArgumentParser(description="Тестирование Wake Word в реальном времени")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Порог вероятности для срабатывания (0.0 - 1.0)")
    parser.add_argument("--russian", action="store_true",
                        help="Использовать русскую модель jarvis_ru вместо базовой модели")
    
    args = parser.parse_args()
    
    test_wake_word_live(args.threshold, args.russian)

if __name__ == "__main__":
    main()