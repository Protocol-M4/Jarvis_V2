import logging
from config import settings

def setup_logger():
    logger = logging.getLogger("Jarvis")
    logger.setLevel(logging.INFO)

    # Формат: Время - Модуль - Уровень - Сообщение
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Лог в файл (путь берем из настроек)
    file_handler = logging.FileHandler(settings.LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Лог в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# Создаем объект логгера для импорта в другие файлы
logger = setup_logger()
