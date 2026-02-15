#!/usr/bin/env python3
import os
import sys
import time
import threading
import warnings
from enum import Enum

# Подавление предупреждения urllib3 о LibreSSL
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal as Signal, pyqtSlot as Slot

# Импортируем компоненты Jarvis
from config import settings
from core.logger import logger
from core.stt_engine import STTEngine
from core.llm_engine import LLMEngine
from core.tts_engine import TTSEngine
from core.wake_word import WakeWordDetector
from core.power_manager import PowerManager
from ui.main_window import MainWindow, JarvisState

class JarvisEngine(QThread):
    """
    Основной движок Jarvis, работающий в отдельном потоке.
    Управляет всеми компонентами системы и обрабатывает запросы пользователя.
    """
    
    # Сигналы для связи с UI
    log_signal = Signal(str)
    status_signal = Signal(object)  # Изменено с Signal(str) для совместимости с update_status
    state_changed = Signal(object)  # JarvisState
    user_message_signal = Signal(str)
    jarvis_message_signal = Signal(str)
    error_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        
        # Флаги состояния
        self.running = False
        self.paused = False
        self.wake_word_active = False
        
        # Текущее состояние
        self.current_state = JarvisState.IDLE
        
        # Инициализация компонентов
        self.init_components()
    
    def init_components(self):
        """Инициализирует все компоненты Jarvis"""
        try:
            self.log_signal.emit("Инициализация компонентов Jarvis...")
            
            # Инициализация STT (Speech-to-Text)
            self.log_signal.emit(f"Загрузка модели Whisper {settings.WHISPER_MODEL_SIZE}...")
            self.ears = STTEngine()
            
            # Инициализация LLM (Language Model)
            self.log_signal.emit("Инициализация языковой модели...")
            self.brain = LLMEngine()
            
            # Инициализация TTS (Text-to-Speech)
            self.log_signal.emit("Инициализация синтеза речи...")
            self.voice = TTSEngine()
            
            # Инициализация детектора ключевого слова
            self.log_signal.emit("Инициализация детектора ключевого слова...")
            self.wake_word_detector = WakeWordDetector()
            
            # Активируем режим ключевого слова по умолчанию
            self.wake_word_active = True
            self.log_signal.emit(f"Режим ключевого слова '{settings.WAKE_WORDS[0]}' активирован по умолчанию")
            
            self.log_signal.emit("Все компоненты инициализированы успешно")
            self.set_state(JarvisState.IDLE)
        except Exception as e:
            self.error_signal.emit(f"Ошибка при инициализации компонентов: {e}")
            logger.error(f"Ошибка при инициализации компонентов: {e}", exc_info=True)
    
    def set_state(self, state):
        """Устанавливает текущее состояние и отправляет сигнал в UI"""
        self.current_state = state
        self.state_changed.emit(state)
        self.status_signal.emit(state)  # Передаем сам объект state вместо state.value
    
    def run(self):
        """Основной цикл работы Jarvis"""
        self.running = True
        
        # Запускаем детектор ключевого слова, если активирован
        if self.wake_word_active:
            self.start_wake_word_detection()
        
        self.log_signal.emit("Jarvis запущен и готов к работе")
        self.voice.say("Я готов к работе, сэр")
        
        # Основной цикл
        while self.running:
            try:
                # Если система на паузе, просто ждем
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Если активирован режим ключевого слова, ждем активации через callback
                if self.wake_word_active:
                    time.sleep(0.1)
                    continue
                
                # Обычный режим работы - слушаем пользователя
                self.set_state(JarvisState.HEARING)
                self.log_signal.emit("Слушаю...")
                
                # Записываем аудио с использованием VAD
                recording_success = self.ears.record_to_file()
                
                # Если запись не удалась (не обнаружена речь), продолжаем слушать
                if not recording_success:
                    self.log_signal.emit("Речь не обнаружена, продолжаю слушать")
                    self.set_state(JarvisState.IDLE)
                    continue
                
                # Преобразуем аудио в текст
                user_text = self.ears.transcribe_file()
                
                if user_text:
                    self.user_message_signal.emit(user_text)
                    logger.info(f"Пользователь: {user_text}")
                    
                    # Проверка на выход
                    if any(word in user_text.lower() for word in ["выход", "отключись", "стоп"]):
                        self.log_signal.emit("Команда завершения принята")
                        self.voice.say("Отключаюсь. Доброй ночи, сэр.")
                        break
                    
                    # Обрабатываем запрос
                    self.set_state(JarvisState.THINKING)
                    self.log_signal.emit("Обрабатываю запрос...")
                    
                    answer = self.brain.ask(user_text)
                    
                    # Защита от глюков API
                    if not answer or len(answer.strip()) < 3:
                        self.log_signal.emit("Получен некорректный ответ, запрашиваю повтор...")
                        answer = self.brain.ask("Повтори еще раз, возникла системная ошибка связи.")
                    
                    # Отвечаем
                    self.set_state(JarvisState.SPEAKING)
                    self.jarvis_message_signal.emit(answer)
                    self.voice.say(answer)
                    
                    # Возвращаемся в режим ожидания
                    self.set_state(JarvisState.IDLE)
                    
                    # Если активирован режим ключевого слова, запускаем его снова
                    if self.wake_word_active:
                        self.start_wake_word_detection()
                else:
                    self.error_signal.emit("Транскрипция не удалась или текст пустой")
                    self.set_state(JarvisState.IDLE)
            
            except Exception as e:
                self.error_signal.emit(f"Ошибка в основном цикле: {e}")
                logger.error(f"Ошибка в основном цикле: {e}", exc_info=True)
                self.set_state(JarvisState.IDLE)
        
        # Завершение работы
        self.log_signal.emit("Завершение работы Jarvis")
        self.stop_wake_word_detection()
    
    def start_wake_word_detection(self):
        """Запускает детектор ключевого слова"""
        try:
            self.log_signal.emit(f"Активация режима ожидания ключевого слова '{settings.WAKE_WORDS[0]}'")
            self.wake_word_detector.start_monitoring(callback=self.on_wake_word_detected)
        except Exception as e:
            self.error_signal.emit(f"Ошибка при запуске детектора ключевого слова: {e}")
            logger.error(f"Ошибка при запуске детектора ключевого слова: {e}", exc_info=True)
    
    def stop_wake_word_detection(self):
        """Останавливает детектор ключевого слова"""
        try:
            if hasattr(self, 'wake_word_detector'):
                self.wake_word_detector.stop_monitoring()
                self.log_signal.emit("Детектор ключевого слова остановлен")
        except Exception as e:
            logger.error(f"Ошибка при остановке детектора ключевого слова: {e}", exc_info=True)
    
    def on_wake_word_detected(self):
        """Callback, вызываемый при обнаружении ключевого слова"""
        self.log_signal.emit(f"Обнаружено ключевое слово '{settings.WAKE_WORDS[0]}'")
        
        # Останавливаем детектор ключевого слова
        self.stop_wake_word_detection()
        
        # Воспроизводим звук активации
        self.voice.say("Да, сэр?")
        
        # Переходим в режим слушания
        self.set_state(JarvisState.HEARING)
        
        # Вместо записи нового аудио, используем уже обрезанный файл
        # Аудио-буфер уже обрезан в WakeWordDetector._check_wake_word и сохранен в settings.TEMP_WAV
        self.log_signal.emit("Транскрибирую существующий аудио-файл с ключевым словом...")
        user_text = self.ears.transcribe_existing_file()
        
        if user_text:
            self.user_message_signal.emit(user_text)
            logger.info(f"Пользователь: {user_text}")
            
            # Проверка на выход
            if any(word in user_text.lower() for word in ["выход", "отключись", "стоп"]):
                self.log_signal.emit("Команда завершения принята")
                self.voice.say("Отключаюсь. Доброй ночи, сэр.")
                self.running = False
                return
            
            # Обрабатываем запрос
            self.set_state(JarvisState.THINKING)
            self.log_signal.emit("Обрабатываю запрос...")
            
            answer = self.brain.ask(user_text)
            
            # Защита от глюков API
            if not answer or len(answer.strip()) < 3:
                self.log_signal.emit("Получен некорректный ответ, запрашиваю повтор...")
                answer = self.brain.ask("Повтори еще раз, возникла системная ошибка связи.")
            
            # Отвечаем
            self.set_state(JarvisState.SPEAKING)
            self.jarvis_message_signal.emit(answer)
            self.voice.say(answer)
            
            # Возвращаемся в режим ожидания
            self.set_state(JarvisState.IDLE)
            
            # Запускаем детектор ключевого слова снова
            self.start_wake_word_detection()
        else:
            self.error_signal.emit("Транскрипция не удалась или текст пустой")
            self.set_state(JarvisState.IDLE)
            
            # Запускаем детектор ключевого слова снова
            self.start_wake_word_detection()
    
    def toggle_wake_word_mode(self, active):
        """Включает или выключает режим ключевого слова"""
        self.wake_word_active = active
        
        if active:
            self.log_signal.emit("Режим ключевого слова включен")
            self.start_wake_word_detection()
        else:
            self.log_signal.emit("Режим ключевого слова выключен")
            self.stop_wake_word_detection()
    
    def set_whisper_model(self, model_name):
        """Изменяет модель Whisper"""
        try:
            # Сохраняем новое значение в настройках
            settings.WHISPER_MODEL_SIZE = model_name
            
            # Перезагружаем STT с новой моделью
            self.log_signal.emit(f"Загрузка модели Whisper {model_name}...")
            self.ears = STTEngine()
            
            self.log_signal.emit(f"Модель Whisper изменена на {model_name}")
        except Exception as e:
            self.error_signal.emit(f"Ошибка при изменении модели Whisper: {e}")
            logger.error(f"Ошибка при изменении модели Whisper: {e}", exc_info=True)
    
    def pause(self):
        """Приостанавливает работу Jarvis"""
        if not self.paused:
            self.paused = True
            self.log_signal.emit("Jarvis приостановлен")
            
            # Останавливаем детектор ключевого слова
            self.stop_wake_word_detection()
    
    def resume(self):
        """Возобновляет работу Jarvis"""
        if self.paused:
            self.paused = False
            self.log_signal.emit("Jarvis возобновил работу")
            
            # Запускаем детектор ключевого слова, если он был активен
            if self.wake_word_active:
                self.start_wake_word_detection()
    
    def stop(self):
        """Останавливает работу Jarvis"""
        if not self.running:
            return  # Уже остановлен
        
        self.log_signal.emit("Останавливаю Jarvis...")
        self.running = False
        
        # Останавливаем детектор ключевого слова с полной деинициализацией
        self.stop_wake_word_detection()
        
        # Останавливаем STT, если он активен
        if hasattr(self, 'ears'):
            try:
                # Если у STTEngine есть активные потоки или ресурсы, останавливаем их
                if hasattr(self.ears, 'stop_recording') and callable(self.ears.stop_recording):
                    self.ears.stop_recording()
            except Exception as e:
                self.error_signal.emit(f"Ошибка при остановке STT: {e}")
                logger.error(f"Ошибка при остановке STT: {e}", exc_info=True)
        
        # Ждем завершения потока
        if self.isRunning():
            self.wait(3000)  # Увеличиваем таймаут до 3 секунд
            if self.isRunning():
                self.log_signal.emit("Принудительное завершение потока Jarvis")
                self.terminate()  # Принудительно завершаем, если не завершился сам
        
        # Освобождаем ресурсы sounddevice
        try:
            import sounddevice as sd
            sd._terminate()  # Принудительно освобождаем все ресурсы sounddevice
            logger.info("Ресурсы sounddevice освобождены")
        except Exception as e:
            logger.error(f"Ошибка при освобождении ресурсов sounddevice: {e}", exc_info=True)
        
        self.log_signal.emit("Jarvis остановлен")

class JarvisApp(QApplication):
    """Основной класс приложения Jarvis"""
    
    def __init__(self, argv):
        super().__init__(argv)
        
        # Создаем главное окно
        self.main_window = MainWindow()
        
        # Создаем движок Jarvis
        self.engine = JarvisEngine()
        
        # Создаем менеджер питания
        self.power_manager = PowerManager(self.engine)
        
        # Соединяем сигналы и слоты
        self.connect_signals()
        
        # Показываем главное окно
        self.main_window.show()
    
    def connect_signals(self):
        """Соединяет сигналы и слоты между компонентами"""
        # Сигналы от движка к UI
        self.engine.log_signal.connect(self.main_window.append_log)
        self.engine.status_signal.connect(self.main_window.update_status)
        self.engine.state_changed.connect(self.main_window.update_state)
        self.engine.user_message_signal.connect(self.main_window.append_user_message)
        self.engine.jarvis_message_signal.connect(self.main_window.append_jarvis_message)
        self.engine.error_signal.connect(self.main_window.append_error)
        
        # Сигналы от UI к движку
        self.main_window.start_button.triggered.connect(self.start_engine)
        self.main_window.stop_button.triggered.connect(self.stop_engine)
        self.main_window.model_combo.currentTextChanged.connect(self.engine.set_whisper_model)
    
    def start_engine(self):
        """Запускает движок Jarvis"""
        if not self.engine.isRunning():
            self.engine.start()
    
    def stop_engine(self):
        """Останавливает движок Jarvis"""
        if self.engine.isRunning():
            self.engine.stop()
    
    def exec(self):
        """Запускает основной цикл приложения"""
        result = super().exec()
        
        # Очистка ресурсов при выходе
        self.cleanup()
        
        return result
    
    def cleanup(self):
        """Очищает ресурсы при выходе"""
        # Останавливаем движок
        if self.engine.isRunning():
            self.engine.stop()
        
        # Очищаем ресурсы менеджера питания
        if hasattr(self, 'power_manager'):
            self.power_manager.cleanup()

def main():
    """Точка входа в приложение"""
    # Настраиваем путь к ресурсам
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    # Создаем и запускаем приложение
    app = JarvisApp(sys.argv)
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())