#!/usr/bin/env python3
"""
Инструмент для инспекции ChromaDB в консоли.
Выводит все записи из базы данных в виде красивой таблицы.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tabulate import tabulate
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from config.settings import MemorySettings

def inspect_database():
    """Подключается к ChromaDB и выводит все записи в виде таблицы."""
    try:
        # Подключаемся к базе данных
        client = PersistentClient(path=MemorySettings.DB_PATH)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MemorySettings.EMBEDDING_MODEL, device="cpu"
        )
        collection = client.get_or_create_collection(
            name=MemorySettings.COLLECTION_NAME, 
            embedding_function=embedding_function
        )
        
        # Получаем все записи
        results = collection.get()
        
        if not results['ids']:
            print("\n🔍 База данных пуста. Воспоминаний пока нет.\n")
            return
        
        # Формируем данные для таблицы
        table_data = []
        for i, (doc_id, document, metadata) in enumerate(zip(
            results['ids'], 
            results['documents'], 
            results['metadatas'] if results['metadatas'] else [None] * len(results['ids'])
        ), 1):
            # Обрезаем длинные тексты для читаемости
            doc_preview = document[:80] + "..." if len(document) > 80 else document
            metadata_str = str(metadata) if metadata else "—"
            
            table_data.append([
                i,
                doc_id[:8] + "...",  # Показываем только начало UUID
                doc_preview,
                metadata_str
            ])
        
        # Выводим таблицу
        print("\n" + "="*100)
        print(f"📚 БАЗА ДАННЫХ ДЖАРВИСА: {MemorySettings.COLLECTION_NAME}")
        print(f"📍 Путь: {MemorySettings.DB_PATH}")
        print(f"📊 Всего записей: {len(results['ids'])}")
        print("="*100 + "\n")
        
        headers = ["№", "ID", "Текст", "Метаданные"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка при чтении базы данных: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    inspect_database()
