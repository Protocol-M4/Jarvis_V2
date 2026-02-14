import os
from dotenv import load_dotenv
from pathlib import Path

# Находим путь к .env относительно текущего файла
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- ПУТИ К ФАЙЛАМ ---
# Используем Path для кроссплатформенности и удобства
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
PROMPTS_DIR = BASE_DIR / "prompts"

# Временные файлы
TEMP_WAV = str(BASE_DIR / "temp_voice.wav")      # Для STT (уши)
TEMP_MP3 = str(BASE_DIR / "temp_output.mp3")     # Для TTS (голос)
LOG_FILE = str(LOGS_DIR / "jarvis.log")

# Файл с личностью Джарвиса
SYSTEM_PROMPT_FILE = str(PROMPTS_DIR / "system_prompt.txt")

# Создаем необходимые папки, если их нет
LOGS_DIR.mkdir(exist_ok=True)
PROMPTS_DIR.mkdir(exist_ok=True)

# --- НАСТРОЙКИ OPENROUTER (МОЗГ) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "Pu-Pu-Pu")
LLM_MODEL = "google/gemini-2.0-flash-001" 
TEMPERATURE = 0.7  # Креативность: 0.1 - робот, 0.9 - сказочник
MAX_TOKENS = 500   # Длина ответа

# --- НАСТРОЙКИ WHISPER (СЛУХ) ---
WHISPER_MODEL_SIZE = "medium" # base, small, tiny (для экономии GPU)
COMPUTE_TYPE = "float32"    # Оптимизация под видеокарту
DEVICE = "cpu"             # Считаем

# --- НАСТРОЙКИ EDGE-TTS (ГОЛОС) ---
TTS_VOICE = "ru-RU-DmitryNeural"

# --- ПАРАМЕТРЫ ЗАПИСИ МИКРОФОНА ---
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
CHANNELS = 1                # Моно

# --- ИНТЕРФЕЙС (Цвета для консоли) ---
class Colors:
    JARVIS = '\033[94m'      # Синий
    USER = '\033[92m'        # Зеленый
    SYSTEM = '\033[93m'      # Желтый
    ERROR = '\033[91m'       # Красный
    END = '\033[0m'

# --- НАСТРОЙКИ ПАМЯТИ (VECTOR DB) ---
class MemorySettings:
    DB_PATH = "data/vector_db"
    COLLECTION_NAME = "jarvis_memory"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Легкая локальная модель

DEBUG_MODE = True
