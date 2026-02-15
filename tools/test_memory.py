#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности MemoryManager.
Проверяет механизм Distance, фильтрацию по важности и связи parent_id.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.memory_manager import MemoryManager
from config.settings import MemorySettings
from core.logger import logger
import time

def test_memory_manager():
    """Тестирует основные функции MemoryManager."""
    print("\n===== ТЕСТИРОВАНИЕ СИСТЕМЫ ПАМЯТИ ДЖАРВИСА =====\n")
    
    # Инициализация MemoryManager
    try:
        memory = MemoryManager()
        print(f"✅ MemoryManager успешно инициализирован")
        print(f"📊 Параметры:")
        print(f"   - База данных: {MemorySettings.DB_PATH}")
        print(f"   - Коллекция: {MemorySettings.COLLECTION_NAME}")
        print(f"   - Модель эмбеддингов: {MemorySettings.EMBEDDING_MODEL}")
        print(f"   - Порог дистанции: {MemorySettings.MEMORY_THRESHOLD}")
    except Exception as e:
        print(f"❌ Ошибка инициализации MemoryManager: {e}")
        return
    
    print("\n----- Тест 1: Фильтрация по важности -----")
    
    # Тест с низкой важностью (должен быть отфильтрован)
    low_importance_fact = "Этот факт имеет низкую важность и не должен сохраняться."
    result = memory.add_fact(
        low_importance_fact, 
        {"test": "importance_filter"}, 
        importance=1
    )
    
    if result is None:
        print("✅ Факт с важностью 1 успешно отфильтрован")
    else:
        print(f"❌ Ошибка: факт с важностью 1 был сохранен с ID: {result}")
    
    # Тест с нормальной важностью (должен сохраниться)
    normal_importance_fact = "Этот факт имеет нормальную важность и должен сохраниться."
    result = memory.add_fact(
        normal_importance_fact, 
        {"test": "importance_filter"}, 
        importance=3
    )
    
    if result:
        print(f"✅ Факт с важностью 3 успешно сохранен с ID: {result[:8]}...")
        fact_id_1 = result
    else:
        print("❌ Ошибка: факт с важностью 3 не был сохранен")
        return
    
    print("\n----- Тест 2: Механизм Distance и Parent ID -----")
    
    # Добавляем первый факт
    original_fact = "Босс любит пить кофе по утрам с молоком и без сахара."
    result = memory.add_fact(
        original_fact, 
        {"test": "distance_mechanism"}, 
        importance=4
    )
    
    if result:
        print(f"✅ Оригинальный факт успешно сохранен с ID: {result[:8]}...")
        original_id = result
    else:
        print("❌ Ошибка: оригинальный факт не был сохранен")
        return
    
    # Небольшая пауза для уверенности
    time.sleep(1)
    
    # Добавляем похожий факт (должен установить parent_id)
    similar_fact = "Босс предпочитает кофе по утрам, обязательно с молоком, но без сахара."
    result = memory.add_fact(
        similar_fact, 
        {"test": "distance_mechanism"}, 
        importance=3
    )
    
    if result:
        print(f"✅ Похожий факт успешно сохранен с ID: {result[:8]}...")
        similar_id = result
    else:
        print("❌ Ошибка: похожий факт не был сохранен")
        return
    
    # Проверяем, установлен ли parent_id
    results = memory.collection.get(ids=[similar_id])
    
    if results and results['metadatas'] and results['metadatas'][0]:
        metadata = results['metadatas'][0]
        if 'parent_id' in metadata and metadata['parent_id'] == original_id:
            print(f"✅ Механизм Distance работает! Установлен parent_id: {metadata['parent_id'][:8]}...")
        else:
            print("❌ Ошибка: parent_id не установлен или установлен неверно")
    else:
        print("❌ Ошибка: не удалось получить метаданные для проверки parent_id")
    
    print("\n----- Тест 3: Получение релевантного контекста -----")
    
    # Тестируем получение контекста
    query = "Что любит пить босс по утрам?"
    documents, distances = memory.get_relevant_context(query, n_results=2)
    
    if documents and distances:
        print(f"✅ Получено {len(documents)} релевантных контекстов:")
        for i, (doc, dist) in enumerate(zip(documents, distances), 1):
            print(f"   {i}. \"{doc[:50]}...\" (релевантность: {1-dist:.2%})")
    else:
        print("❌ Ошибка: не удалось получить релевантный контекст")
    
    print("\n===== ТЕСТИРОВАНИЕ ЗАВЕРШЕНО =====\n")

if __name__ == "__main__":
    test_memory_manager()