import os
import logging
import uuid
from chromadb import PersistentClient, Settings
from chromadb.utils import embedding_functions
from config.settings import MemorySettings
from core.logger import logger

class MemoryManager:
    def __init__(self):
        try:
            self.client = PersistentClient(path=MemorySettings.DB_PATH, settings=Settings(allow_reset=True))
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=MemorySettings.EMBEDDING_MODEL, device="cpu"
            )
            self.collection = self.client.get_or_create_collection(
                name=MemorySettings.COLLECTION_NAME, embedding_function=self.embedding_function
            )
            logger.info(f"Connected to ChromaDB at {MemorySettings.DB_PATH}, collection name: {MemorySettings.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Error initializing MemoryManager: {e}")
            raise

    def add_fact(self, text: str, metadata: dict = None):
        if not isinstance(text, str):
            logger.error("Fact must be a string.")
            return None
        if metadata and not isinstance(metadata, dict):
            logger.error("Metadata must be a dictionary.")
            return None
        try:
            fact_id = str(uuid.uuid4())
            self.collection.add(
                documents=[text],
                ids=[fact_id],
                metadatas=[metadata] if metadata else None
            )
            logger.info(f"Added fact with id {fact_id}: {text[:50]}...")  # Log first 50 chars
            return fact_id
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
            return None

    def get_relevant_context(self, query: str, n_results: int = 3):
        if not isinstance(query, str):
            logger.error("Query must be a string.")
            return None
        if not isinstance(n_results, int) or n_results <= 0:
            logger.error("Number of results must be a positive integer.")
            return None
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            logger.info(f"Retrieved {n_results} relevant contexts for query: {query[:50]}...")
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return []

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