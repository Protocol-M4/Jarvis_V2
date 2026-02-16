import os
import sys
from enum import Enum
from PyQt6.QtWidgets import (
    QMainWindow, QTextEdit, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QStatusBar, QSplitter,
    QApplication, QStyle, QToolBar, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot, QTimer
from PyQt6.QtGui import QIcon, QTextCursor, QColor, QAction, QFont

# Магия импорта: позволяем Python видеть папку config, которая на уровень выше
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.logger import logger

class JarvisState(Enum):
    """Перечисление возможных состояний Джарвиса"""
    IDLE = "Ожидание активации"
    HEARING = "Слушаю..."
    THINKING = "Обрабатываю запрос..."
    SPEAKING = "Говорю..."

class LogWidget(QTextEdit):
    """Виджет для отображения логов с цветовой подсветкой"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")
        
        # Цвета для разных типов сообщений
        self.colors = {
            "user": QColor("#4EC9B0"),    # Бирюзовый для пользователя
            "jarvis": QColor("#569CD6"),  # Синий для Джарвиса
            "system": QColor("#DCDCAA"),  # Желтый для системных сообщений
            "error": QColor("#F44747"),   # Красный для ошибок
            "info": QColor("#9CDCFE"),    # Светло-синий для информации
            "debug": QColor("#6A9955")    # Зеленый для отладки
        }
    
    def append_message(self, text, msg_type="system"):
        """
        Добавляет сообщение в лог с соответствующим цветом
        
        Args:
            text: Текст сообщения
            msg_type: Тип сообщения (user, jarvis, system, error, info, debug)
        """
        color = self.colors.get(msg_type, QColor("#FFFFFF"))
        
        # Сохраняем текущую позицию курсора
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Устанавливаем цвет текста
        format = cursor.charFormat()
        format.setForeground(color)
        cursor.setCharFormat(format)
        
        # Добавляем текст
        cursor.insertText(text + "\n")
        
        # Прокручиваем до конца
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

class MainWindow(QMainWindow):
    """Главное окно приложения Jarvis"""
    
    def __init__(self):
        super().__init__()
        
        # Настройка окна
        self.setWindowTitle("Jarvis Assistant")
        self.setMinimumSize(800, 600)
        
        # Создаем центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Основной layout
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Создаем виджет для логов
        self.log_widget = LogWidget()
        
        # Создаем панель статуса
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Создаем панель инструментов
        self.toolbar = QToolBar("Управление")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        
        # Добавляем кнопки управления
        self.start_button = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Запустить", self)
        self.start_button.setStatusTip("Запустить Джарвиса")
        self.start_button.triggered.connect(self.on_start)
        self.toolbar.addAction(self.start_button)
        
        self.stop_button = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Остановить", self)
        self.stop_button.setStatusTip("Остановить Джарвиса")
        self.stop_button.triggered.connect(self.on_stop)
        self.toolbar.addAction(self.stop_button)
        
        self.toolbar.addSeparator()
        
        # Добавляем выпадающий список для выбора модели Whisper
        self.model_label = QLabel("Модель Whisper:")
        self.toolbar.addWidget(self.model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText(settings.WHISPER_MODEL_SIZE)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.toolbar.addWidget(self.model_combo)
        
        # Добавляем лог в основной layout
        self.main_layout.addWidget(self.log_widget)
        
        # Инициализируем состояние
        self.update_status(JarvisState.IDLE)
        
        # Приветственное сообщение
        self.log_widget.append_message("Jarvis Assistant запущен", "system")
        self.log_widget.append_message("Ожидание инициализации...", "system")
    
    @Slot()
    def on_start(self):
        """Обработчик нажатия кнопки Запустить"""
        self.log_widget.append_message("Запуск Джарвиса...", "system")
        # Сигнал будет подключен к JarvisEngine
    
    @Slot()
    def on_stop(self):
        """Обработчик нажатия кнопки Остановить"""
        self.log_widget.append_message("Остановка Джарвиса...", "system")
        
        # Блокируем обе кнопки на время остановки
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        # Получаем доступ к приложению и останавливаем движок
        app = QApplication.instance()
        if hasattr(app, 'engine'):
            app.engine.stop()
        
        # Разблокируем кнопку "Старт" через 2 секунды
        QTimer.singleShot(2000, lambda: self.start_button.setEnabled(True))
    
    @Slot(str)
    def on_model_changed(self, model_name):
        """Обработчик изменения модели Whisper"""
        self.log_widget.append_message(f"Выбрана модель Whisper: {model_name}", "system")
        # Сигнал будет подключен к JarvisEngine
    
    @Slot(str)
    def append_log(self, text):
        """Добавляет текст в лог"""
        self.log_widget.append_message(text, "info")
    
    def append_user_message(self, text):
        """Добавляет сообщение пользователя в лог"""
        self.log_widget.append_message(f"Вы: {text}", "user")
    
    def append_jarvis_message(self, text):
        """Добавляет ответ Джарвиса в лог"""
        self.log_widget.append_message(f"Джарвис: {text}", "jarvis")
    
    def append_error(self, text):
        """Добавляет сообщение об ошибке в лог"""
        self.log_widget.append_message(f"Ошибка: {text}", "error")
    
    @Slot(JarvisState)
    def update_state(self, state):
        """Обновляет состояние Джарвиса"""
        self.update_status(state.value)
        
        # Обновляем UI в зависимости от состояния
        if state == JarvisState.IDLE:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            # Сбрасываем стиль статус-бара
            self.status_bar.setStyleSheet("")
        elif state == JarvisState.HEARING:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            # Подсвечиваем статус-бар при активации слушания (особенно после wake word)
            self.status_bar.setStyleSheet("background-color: #4EC9B0; color: black; font-weight: bold;")
        else:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            # Сбрасываем стиль статус-бара для других состояний
            self.status_bar.setStyleSheet("")
    
    @Slot(object)  # Изменяем аннотацию, чтобы принимать разные типы
    def update_status(self, status):
        """Обновляет строку статуса"""
        # Проверяем тип status и преобразуем при необходимости
        if isinstance(status, JarvisState):
            status_text = status.value
        else:
            status_text = str(status)
        self.status_bar.showMessage(status_text)
    
    def closeEvent(self, event):
        """Обработчик закрытия окна"""
        self.log_widget.append_message("Завершение работы Джарвиса...", "system")
        
        # Получаем доступ к родительскому приложению и останавливаем движок
        app = QApplication.instance()
        if hasattr(app, 'engine') and app.engine.isRunning():
            self.log_widget.append_message("Останавливаю Jarvis Engine...", "system")
            app.engine.stop()
        
        event.accept()

# Для тестирования
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    # Тестовые сообщения
    window.append_log("Система инициализирована")
    window.append_user_message("Привет, Джарвис!")
    window.append_jarvis_message("Здравствуйте! Чем я могу вам помочь?")
    window.append_error("Не удалось подключиться к серверу")
    
    sys.exit(app.exec())