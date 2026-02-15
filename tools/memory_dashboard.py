#!/usr/bin/env python3
"""
Веб-дашборд для просмотра памяти Джарвиса на базе Streamlit.
Позволяет просматривать все воспоминания и искать по ним.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from config.settings import MemorySettings
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Память Джарвиса",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def get_collection():
    """Подключается к ChromaDB и возвращает коллекцию."""
    client = PersistentClient(path=MemorySettings.DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MemorySettings.EMBEDDING_MODEL, device="cpu"
    )
    collection = client.get_or_create_collection(
        name=MemorySettings.COLLECTION_NAME, 
        embedding_function=embedding_function
    )
    return collection

def main():
    # Заголовок
    st.title("🧠 Память Джарвиса")
    st.markdown("---")
    
    # Получаем коллекцию
    try:
        collection = get_collection()
    except Exception as e:
        st.error(f"❌ Ошибка подключения к базе данных: {e}")
        return
    
    # Получаем все записи
    try:
        all_results = collection.get()
        total_memories = len(all_results['ids'])
    except Exception as e:
        st.error(f"❌ Ошибка при чтении данных: {e}")
        return
    
    # Информационная панель
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Всего воспоминаний", total_memories)
    with col2:
        st.metric("📁 Коллекция", MemorySettings.COLLECTION_NAME)
    with col3:
        st.metric("🗄️ База данных", MemorySettings.DB_PATH)
    
    st.markdown("---")
    
    # Поиск по воспоминаниям
    st.subheader("🔍 Поиск по воспоминаниям")
    search_query = st.text_input(
        "Введите запрос для семантического поиска:",
        placeholder="Например: что ты знаешь о моих предпочтениях?"
    )
    
    n_results = st.slider("Количество результатов:", min_value=1, max_value=10, value=5)
    
    if search_query:
        try:
            search_results = collection.query(
                query_texts=[search_query],
                n_results=min(n_results, total_memories) if total_memories > 0 else 1
            )
            
            st.markdown("### 🎯 Результаты поиска:")
            
            if search_results['ids'][0]:
                for i, (doc_id, document, distance, metadata) in enumerate(zip(
                    search_results['ids'][0],
                    search_results['documents'][0],
                    search_results['distances'][0],
                    search_results['metadatas'][0] if search_results['metadatas'] else [None] * len(search_results['ids'][0])
                ), 1):
                    with st.expander(f"📝 Результат #{i} (релевантность: {1 - distance:.2%})"):
                        st.write(f"**Текст:** {document}")
                        st.write(f"**ID:** `{doc_id}`")
                        if metadata:
                            st.write(f"**Метаданные:** {metadata}")
                        st.write(f"**Дистанция:** {distance:.4f}")
            else:
                st.info("Ничего не найдено.")
                
        except Exception as e:
            st.error(f"❌ Ошибка поиска: {e}")
    
    st.markdown("---")
    
    # Список всех воспоминаний
    st.subheader("📚 Все воспоминания")
    
    if total_memories == 0:
        st.info("🔍 База данных пуста. Воспоминаний пока нет.")
    else:
        # Фильтр по количеству отображаемых записей
        show_all = st.checkbox("Показать все записи", value=False)
        display_limit = total_memories if show_all else min(20, total_memories)
        
        if not show_all and total_memories > 20:
            st.info(f"Показаны первые {display_limit} из {total_memories} записей. Включите 'Показать все записи' для полного списка.")
        
        # Отображаем записи
        for i, (doc_id, document, metadata) in enumerate(zip(
            all_results['ids'][:display_limit],
            all_results['documents'][:display_limit],
            all_results['metadatas'][:display_limit] if all_results['metadatas'] else [None] * display_limit
        ), 1):
            with st.expander(f"💭 Воспоминание #{i}"):
                st.write(f"**Текст:** {document}")
                st.write(f"**ID:** `{doc_id}`")
                if metadata:
                    st.write(f"**Метаданные:** {metadata}")
    
    # Футер
    st.markdown("---")
    st.caption(f"🤖 Jarvis Memory Dashboard | Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")

if __name__ == "__main__":
    main()
