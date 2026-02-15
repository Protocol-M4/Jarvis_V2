import os
import logging
import uuid
from datetime import datetime
from chromadb import PersistentClient, Settings
from chromadb.utils import embedding_functions
from config.settings import MemorySettings
from core.logger import logger

class MemoryManager:
    def __init__(self):
        try:
            self.client = PersistentClient(path=MemorySettings.DB_PATH, settings=Settings(allow_reset=True))
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=MemorySettings.EMBEDDING_MODEL, device="mps"  # Используем GPU на M4 Pro
            )
            self.collection = self.client.get_or_create_collection(
                name=MemorySettings.COLLECTION_NAME, embedding_function=self.embedding_function
            )
            logger.info(f"Connected to ChromaDB at {MemorySettings.DB_PATH}, collection name: {MemorySettings.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Error initializing MemoryManager: {e}")
            raise

    def add_fact(self, text: str, metadata: dict = None, importance: int = 3):
        # Проверка на важность (фильтр)
        if importance == 1:
            logger.info(f"Факт проигнорирован из-за низкой важности: {text[:50]}...")
            return None
            
        # Проверки входных данных
        if not isinstance(text, str):
            logger.error("Fact must be a string.")
            return None
        if metadata and not isinstance(metadata, dict):
            logger.error("Metadata must be a dictionary.")
            return None
            
        # Генерируем ID для новой записи
        fact_id = str(uuid.uuid4())
        
        try:
            # Проверка на схожесть с существующими записями
            results = self.collection.query(
                query_texts=[text],
                n_results=1
            )
            
            # Логирование полной структуры результатов для отладки
            logger.debug(f"ChromaDB query results: {results}")
            
            # Более надежная проверка наличия результатов
            if (results.get('documents') and results.get('distances') and 
                len(results['documents']) > 0 and len(results['documents'][0]) > 0 and
                len(results['distances']) > 0 and len(results['distances'][0]) > 0 and
                results['distances'][0][0] < MemorySettings.MEMORY_THRESHOLD):
                
                # Проверка наличия ID
                if results.get('ids') and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                    parent_id = results['ids'][0][0]
                    
                    # Добавляем parent_id в метаданные
                    if metadata is None:
                        metadata = {}
                    metadata['parent_id'] = parent_id
                    
                    logger.info(f"Найден похожий факт (distance: {results['distances'][0][0]:.4f}), " 
                               f"установлена связь parent_id: {parent_id}")
            
            # Добавляем timestamp, если его нет
            if metadata is None:
                metadata = {}
            if 'timestamp' not in metadata:
                metadata['timestamp'] = datetime.now().isoformat()
                
            # Добавляем запись в коллекцию
            logger.debug(f"Adding fact to ChromaDB: text='{text[:100]}...', id={fact_id}")
            add_result = self.collection.add(
                documents=[text],
                ids=[fact_id],
                metadatas=[metadata] if metadata else None
            )
            
            # Логируем результат операции добавления
            logger.debug(f"ChromaDB add result: {add_result}")
            
            logger.info(f"Добавлен факт с id {fact_id}: {text[:50]}...")
            return fact_id
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
            # Добавляем более подробную информацию об ошибке
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
            return None

    def get_relevant_context(self, query: str, n_results: int = 3):
        if not isinstance(query, str):
            logger.error("Query must be a string.")
            return [], []
        if not isinstance(n_results, int) or n_results <= 0:
            logger.error("Number of results must be a positive integer.")
            return [], []
        try:
            logger.debug(f"Querying ChromaDB for relevant context: query='{query[:100]}...', n_results={n_results}")
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Логирование полной структуры результатов для отладки
            logger.debug(f"ChromaDB query results: {results}")
            
            # Более надежная проверка наличия результатов
            documents = []
            distances = []
            
            if results.get('documents') and len(results['documents']) > 0:
                documents = results['documents'][0]
                logger.info(f"Retrieved {len(documents)} relevant contexts for query: {query[:50]}...")
            else:
                logger.info(f"No relevant contexts found for query: {query[:50]}...")
                
            if results.get('distances') and len(results['distances']) > 0:
                distances = results['distances'][0]
                
            return documents, distances
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            # Добавляем более подробную информацию об ошибке
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
            return [], []

if __name__ == '__main__':
    # Example usage (for testing purposes)
    try:
        memory_manager = MemoryManager()
        fact_id = memory_manager.add_fact("The sky is blue.", {"source": "observation"})
        if fact_id:
            print(f"Fact added with ID: {fact_id}")
            relevant_context = memory_manager.get_relevant_context("What color is the sky?", n_results=1)
            if relevant_context:
                print(f"Relevant context: {relevant_context}")
            else:
                print("No relevant context found.")
        else:
            print("Failed to add fact.")
    except Exception as e:
        print(f"An error occurred: {e}")