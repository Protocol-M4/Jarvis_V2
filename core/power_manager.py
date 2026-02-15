import os
import sys
import logging
import threading

# Магия импорта: позволяем Python видеть папку config, которая на уровень выше
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger

# Импортируем AppKit только если мы на macOS
try:
    import AppKit
    import objc
    MACOS_AVAILABLE = True
except ImportError:
    logger.warning("AppKit не найден. Интеграция с macOS не будет работать.")
    MACOS_AVAILABLE = False

class PowerManager:
    """
    Класс для управления энергопотреблением и обработки событий сна/пробуждения системы.
    В настоящее время поддерживается только macOS через AppKit.
    """
    
    def __init__(self, jarvis_engine=None):
        """
        Инициализация менеджера питания
        
        Args:
            jarvis_engine: Экземпляр движка Jarvis, который будет уведомлен о событиях сна/пробуждения
        """
        self.jarvis_engine = jarvis_engine
        self.observers_registered = False
        
        # Флаг для отслеживания состояния системы
        self.system_sleeping = False
        
        if MACOS_AVAILABLE:
            self._setup_macos_observers()
        else:
            logger.warning("Интеграция с управлением питанием недоступна на этой платформе")
    
    def _setup_macos_observers(self):
        """Настройка наблюдателей за событиями сна/пробуждения в macOS"""
        try:
            # Получаем общее рабочее пространство
            self.workspace = AppKit.NSWorkspace.sharedWorkspace()
            self.nc = self.workspace.notificationCenter()
            
            # Регистрируем наблюдателей
            self.nc.addObserver_selector_name_object_(
                self,
                objc.selector(self.sleep_notification, signature=b'v@:@'),
                AppKit.NSWorkspaceWillSleepNotification,
                None
            )
            
            self.nc.addObserver_selector_name_object_(
                self,
                objc.selector(self.wake_notification, signature=b'v@:@'),
                AppKit.NSWorkspaceDidWakeNotification,
                None
            )
            
            # Также отслеживаем выход из системы и выключение
            self.nc.addObserver_selector_name_object_(
                self,
                objc.selector(self.shutdown_notification, signature=b'v@:@'),
                AppKit.NSWorkspaceWillPowerOffNotification,
                None
            )
            
            self.observers_registered = True
            logger.info("Наблюдатели за событиями питания macOS зарегистрированы")
        except Exception as e:
            logger.error(f"Ошибка при настройке наблюдателей macOS: {e}")
    
    def sleep_notification(self, notification):
        """
        Обработчик уведомления о переходе в режим сна
        
        Args:
            notification: Объект уведомления от macOS
        """
        logger.info("Система переходит в режим сна")
        self.system_sleeping = True
        
        # Уведомляем движок Jarvis
        if self.jarvis_engine:
            try:
                # Запускаем в отдельном потоке с таймаутом, чтобы не блокировать переход в сон
                thread = threading.Thread(target=self._safe_pause)
                thread.daemon = True
                thread.start()
                thread.join(timeout=2.0)  # Даем максимум 2 секунды на остановку
            except Exception as e:
                logger.error(f"Ошибка при обработке события сна: {e}")
    
    def wake_notification(self, notification):
        """
        Обработчик уведомления о выходе из режима сна
        
        Args:
            notification: Объект уведомления от macOS
        """
        logger.info("Система вышла из режима сна")
        self.system_sleeping = False
        
        # Уведомляем движок Jarvis
        if self.jarvis_engine:
            try:
                # Запускаем в отдельном потоке, чтобы не блокировать UI
                thread = threading.Thread(target=self._safe_resume)
                thread.daemon = True
                thread.start()
            except Exception as e:
                logger.error(f"Ошибка при обработке события пробуждения: {e}")
    
    def shutdown_notification(self, notification):
        """
        Обработчик уведомления о выключении системы
        
        Args:
            notification: Объект уведомления от macOS
        """
        logger.info("Система выключается")
        
        # Уведомляем движок Jarvis
        if self.jarvis_engine:
            try:
                # Запускаем в отдельном потоке с таймаутом
                thread = threading.Thread(target=self._safe_shutdown)
                thread.daemon = True
                thread.start()
                thread.join(timeout=2.0)  # Даем максимум 2 секунды на остановку
            except Exception as e:
                logger.error(f"Ошибка при обработке события выключения: {e}")
    
    def _safe_pause(self):
        """Безопасная остановка Jarvis с обработкой исключений"""
        try:
            if hasattr(self.jarvis_engine, 'pause'):
                self.jarvis_engine.pause()
            else:
                logger.warning("Метод pause не найден в движке Jarvis")
        except Exception as e:
            logger.error(f"Ошибка при вызове pause: {e}")
    
    def _safe_resume(self):
        """Безопасное возобновление работы Jarvis с обработкой исключений"""
        try:
            if hasattr(self.jarvis_engine, 'resume'):
                self.jarvis_engine.resume()
            else:
                logger.warning("Метод resume не найден в движке Jarvis")
        except Exception as e:
            logger.error(f"Ошибка при вызове resume: {e}")
    
    def _safe_shutdown(self):
        """Безопасное завершение работы Jarvis с обработкой исключений"""
        try:
            if hasattr(self.jarvis_engine, 'shutdown'):
                self.jarvis_engine.shutdown()
            elif hasattr(self.jarvis_engine, 'stop'):
                self.jarvis_engine.stop()
            else:
                logger.warning("Методы shutdown/stop не найдены в движке Jarvis")
        except Exception as e:
            logger.error(f"Ошибка при вызове shutdown/stop: {e}")
    
    def cleanup(self):
        """Очистка ресурсов и отмена регистрации наблюдателей"""
        if MACOS_AVAILABLE and self.observers_registered:
            try:
                # Отменяем регистрацию наблюдателей
                self.nc.removeObserver_(self)
                logger.info("Наблюдатели за событиями питания macOS удалены")
            except Exception as e:
                logger.error(f"Ошибка при удалении наблюдателей macOS: {e}")