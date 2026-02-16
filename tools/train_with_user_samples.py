#!/usr/bin/env python3
"""
Скрипт для записи пользовательских сэмплов, обучения модели с этими сэмплами
и тестирования новой модели. Объединяет все шаги в один процесс.
"""

import os
import sys
import argparse
import subprocess
import time

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def run_command(command, description):
    """
    Запускает команду и выводит ее результат.
    
    Args:
        command (list): Команда для запуска
        description (str): Описание команды
    """
    print(f"\n=== {description} ===")
    print(f"Выполняем: {' '.join(command)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении команды: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Основная функция скрипта"""
    parser = argparse.ArgumentParser(description="Запись сэмплов, обучение и тестирование модели")
    parser.add_argument("--samples", type=int, default=15,
                        help="Количество сэмплов для записи")
    parser.add_argument("--weight", type=int, default=75,
                        help="Вес пользовательских сэмплов (количество дублирований)")
    parser.add_argument("--dataset", default="data/wake_word_dataset_ru",
                        help="Директория с основным датасетом")
    parser.add_argument("--user-samples", default="data/my_samples",
                        help="Директория для сохранения пользовательских сэмплов")
    parser.add_argument("--model-name", default="jarvis_ru_universal",
                        help="Имя модели")
    parser.add_argument("--skip-recording", action="store_true",
                        help="Пропустить запись сэмплов (использовать существующие)")
    
    args = parser.parse_args()
    
    # Шаг 1: Запись пользовательских сэмплов
    if not args.skip_recording:
        print("\n=== Шаг 1: Запись пользовательских сэмплов ===")
        record_command = [
            "python", "tools/record_samples.py",
            "--output", args.user_samples,
            "--samples", str(args.samples)
        ]
        if not run_command(record_command, "Запись пользовательских сэмплов"):
            print("Ошибка при записи сэмплов. Прерываем процесс.")
            return False
    else:
        print("\n=== Шаг 1: Пропускаем запись сэмплов ===")
        # Проверяем наличие сэмплов
        if not os.path.exists(args.user_samples):
            print(f"Ошибка: директория {args.user_samples} не существует")
            return False
        
        sample_files = [f for f in os.listdir(args.user_samples) if f.endswith('.wav')]
        if not sample_files:
            print(f"Ошибка: в директории {args.user_samples} нет WAV файлов")
            return False
        
        print(f"Найдено {len(sample_files)} существующих сэмплов в {args.user_samples}")
    
    # Шаг 2: Обучение модели с пользовательскими сэмплами
    print("\n=== Шаг 2: Обучение модели с пользовательскими сэмплами ===")
    train_command = [
        "python", "tools/train_model.py",
        "--dataset", args.dataset,
        "--name", args.model_name,
        "--user-samples", args.user_samples,
        "--user-weight", str(args.weight)
    ]
    if not run_command(train_command, "Обучение модели"):
        print("Ошибка при обучении модели. Прерываем процесс.")
        return False
    
    # Шаг 3: Тестирование новой модели
    print("\n=== Шаг 3: Тестирование новой модели ===")
    print("Запускаем тестирование модели. Говорите 'Джарвис' для проверки.")
    print("Нажмите Ctrl+C для завершения тестирования.")
    print("=" * 50)
    
    # Даем пользователю время подготовиться
    print("Тестирование начнется через 3 секунды...")
    time.sleep(3)
    
    # Запускаем тестирование
    test_command = [
        "python", "tools/test_wake_word_live.py",
        "--russian"
    ]
    
    try:
        # Запускаем тестирование без перехвата вывода, чтобы пользователь видел результаты в реальном времени
        subprocess.run(test_command, check=False)
    except KeyboardInterrupt:
        print("\nТестирование прервано пользователем.")
    
    print("\n=== Процесс завершен ===")
    print(f"Модель {args.model_name} обучена с использованием пользовательских сэмплов")
    print(f"Вес пользовательских сэмплов: {args.weight}")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    main()