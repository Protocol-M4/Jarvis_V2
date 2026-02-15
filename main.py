#!/usr/bin/env python3
"""
Jarvis Assistant - Голосовой помощник с Wake Word активацией
Версия 2.0 - Автономное приложение с графическим интерфейсом

Этот файл является обратно-совместимым запускателем для консольной версии.
Для запуска полноценного приложения используйте app.py
"""

import os
import sys
import subprocess
import warnings

# Подавление предупреждения urllib3 о LibreSSL
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

from core.logger import logger

def main():
    """
    Запускает новую версию Jarvis Assistant через app.py
    """
    logger.info("Запуск Jarvis Assistant v2.0...")
    
    # Определяем путь к app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "app.py")
    
    # Проверяем наличие файла
    if not os.path.exists(app_path):
        logger.error(f"Файл {app_path} не найден")
        print(f"Ошибка: Файл {app_path} не найден")
        return 1
    
    try:
        # Запускаем новое приложение
        logger.info(f"Запуск {app_path}...")
        print("Запуск Jarvis Assistant v2.0...")
        
        # Делаем файл исполняемым (для Unix-подобных систем)
        if sys.platform != "win32":
            os.chmod(app_path, 0o755)
        
        # Запускаем приложение
        result = subprocess.call([sys.executable, app_path] + sys.argv[1:])
        return result
    
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {e}", exc_info=True)
        print(f"Ошибка при запуске приложения: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
