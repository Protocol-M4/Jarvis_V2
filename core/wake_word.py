import os
import sys
import time
import queue
import threading
import numpy as np
import torch
import sounddevice as sd
from collections import deque
from openwakeword import Model

# Магия импорта: позволяем Python видеть папку config, которая на уровень выше
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.logger import logger

class WakeWordDetector:
    """
    Класс для обнаружения ключевого слова (wake word) с использованием
    OpenWakeWord - легковесной и быстрой модели для детекции ключевых слов
    """
    
    def __init__(self):
        """Инициализация детектора ключевого слова"""
        logger.info("Инициализация системы обнаружения ключевого слова...")
        
        # Инициализация Silero VAD для предварительной фильтрации
        logger.info(f"Загрузка модели Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              trust_repo=True)
        
        # Получаем только нужные функции из utils
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        
        # Переводим модель в режим оценки
        self.vad_model.eval()
        
        # Инициализация OpenWakeWord
        logger.info("Загрузка модели OpenWakeWord...")
        
        # Загружаем предобученную модель "hey_jarvis"
        logger.info("Загружаем предобученную модель 'hey_jarvis'")
        self.oww_model = Model(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx"
        )
        
        # Проверяем наличие пользовательской модели (для логирования)
        model_path = os.path.join("models", "openwakeword", "jarvis_ru_universal.joblib")
        if os.path.exists(model_path):
            logger.info(f"Пользовательская модель найдена: {model_path}, но используется базовая модель hey_jarvis")
            
            # Пробуем загрузить модель через joblib для проверки
            try:
                import joblib
                model_data = joblib.load(model_path)
                logger.info(f"Модель успешно загружена через joblib: {type(model_data)}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели через joblib: {e}")
        else:
            logger.info("Пользовательская модель не найдена, используется предобученная 'hey_jarvis'")
        
        # Параметры
        self.sample_rate = settings.SAMPLE_RATE
        self.speech_threshold = settings.WAKE_WORD_THRESHOLD
        self.buffer_size = int(self.sample_rate * 3)  # 3 секунды буфера для OpenWakeWord
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Буфер для VAD (гарантированно 512 сэмплов при 16кГц)
        self.vad_buffer_size = int(16000 / 16000 * 512)  # 512 сэмплов при 16кГц
        self.vad_buffer = deque(maxlen=self.vad_buffer_size)
        
        # Флаги состояния
        self.is_running = False
        self.is_muted = False  # Флаг для временного отключения обработки
        self.mute_until = 0    # Время, до которого детектор будет отключен
        
        logger.info("Система обнаружения ключевого слова инициализирована")
    
    def start_monitoring(self, callback=None):
        """
        Запускает постоянный мониторинг аудио для обнаружения ключевого слова
        
        Args:
            callback: функция, которая будет вызвана при обнаружении ключевого слова
        """
        if self.is_running:
            logger.warning("Мониторинг уже запущен")
            return
        
        self.is_running = True
        self.callback = callback
        
        # Запускаем мониторинг в отдельном потоке
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Мониторинг ключевого слова запущен")
    
    def stop_monitoring(self):
        """Останавливает мониторинг аудио"""
        if not self.is_running:
            return
        
        logger.info("Останавливаю мониторинг ключевого слова...")
        self.is_running = False
        
        # Отправляем poison pill в очередь, если она существует
        if hasattr(self, 'audio_queue'):
            try:
                # Очищаем очередь перед отправкой poison pill
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.task_done()
                    except:
                        pass
                
                # Отправляем poison pill
                self.audio_queue.put(None)
            except Exception as e:
                logger.error(f"Ошибка при очистке очереди: {e}", exc_info=True)
        
        # Останавливаем и закрываем аудио-стрим
        if hasattr(self, 'stream'):
            try:
                logger.info("Останавливаю аудио-стрим...")
                self.stream.stop()
                self.stream.close()
                logger.info("Аудио-стрим остановлен и закрыт")
            except Exception as e:
                logger.error(f"Ошибка при закрытии аудио-стрима: {e}", exc_info=True)
        
        # Ждем завершения потока
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            logger.info("Ожидаю завершения потока мониторинга...")
            self.monitor_thread.join(timeout=2.0)
            if self.monitor_thread.is_alive():
                logger.warning("Поток мониторинга не завершился корректно")
        
        # Очищаем буферы
        self.audio_buffer.clear()
        self.vad_buffer.clear()
        
        logger.info("Мониторинг ключевого слова остановлен")
    
    def mute(self, duration_seconds=2.0):
        """
        Временно отключает обработку аудио на указанное количество секунд.
        Используется для предотвращения самоактивации при воспроизведении звука.
        
        Args:
            duration_seconds (float): Длительность отключения в секундах
        """
        self.is_muted = True
        self.mute_until = time.time() + duration_seconds
        logger.info(f"Детектор ключевого слова отключен на {duration_seconds} секунд")
        print(f"{settings.Colors.SYSTEM}[WAKE] Детектор ключевого слова отключен на {duration_seconds} секунд{settings.Colors.END}")
    
    def _monitoring_loop(self):
        """Основной цикл мониторинга аудио"""
        # Очищаем буферы перед началом
        self.audio_buffer.clear()
        self.vad_buffer.clear()
        
        # Логирование начала цикла мониторинга
        logger.info("Запущен цикл мониторинга ключевого слова")
        print(f"{settings.Colors.SYSTEM}[WAKE] Запущен цикл мониторинга ключевого слова '{settings.WAKE_WORDS[0]}'{settings.Colors.END}")
        
        # Создаем очередь для обмена данными между потоками
        self.audio_queue = queue.Queue()
        stop_event = threading.Event()
        
        # Функция для записи аудио в отдельном потоке
        def audio_callback(indata, frames, time, status):
            if status:
                logger.error(f"Ошибка записи: {status}")
            # Преобразуем данные в формат, подходящий для обработки
            audio_chunk = indata.copy().flatten()
            if self.is_running:  # Проверяем флаг перед добавлением в очередь
                self.audio_queue.put(audio_chunk)
        
        # Функция обработки аудио в отдельном потоке
        def audio_processing():
            # Буфер для накопления 512 сэмплов для VAD
            vad_chunk_size = int(self.sample_rate / 16000 * 512)  # Размер чанка для VAD при текущей частоте дискретизации
            speech_frames_count = 0
            
            while not stop_event.is_set() and self.is_running:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    # Проверяем на poison pill
                    if audio_chunk is None:
                        logger.info("Получен сигнал завершения (poison pill)")
                        break
                    
                    # Добавляем чанк в основной буфер
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Проверяем, не истекло ли время отключения
                    if self.is_muted and time.time() > self.mute_until:
                        self.is_muted = False
                        logger.info("Детектор ключевого слова снова активен")
                        print(f"{settings.Colors.SYSTEM}[WAKE] Детектор ключевого слова снова активен{settings.Colors.END}")
                    
                    # Если детектор отключен, пропускаем обработку
                    if self.is_muted:
                        self.audio_queue.task_done()
                        # Добавляем небольшую задержку для снижения нагрузки на CPU
                        time.sleep(0.01)
                        continue
                    
                    # Добавляем данные в VAD буфер и обрабатываем, когда накопится нужное количество
                    for sample in audio_chunk:
                        self.vad_buffer.append(sample)
                        
                        # Когда VAD буфер заполнен, обрабатываем его
                        if len(self.vad_buffer) == self.vad_buffer_size:
                            # Конвертируем VAD буфер в torch tensor
                            vad_audio = np.array(list(self.vad_buffer))
                            vad_tensor = torch.tensor(vad_audio, dtype=torch.float32)
                            
                            # Прямой вызов VAD модели
                            speech_prob = self.vad_model(vad_tensor, 16000).item()
                            
                            if speech_prob >= self.speech_threshold:
                                # Речь обнаружена, увеличиваем счетчик
                                speech_frames_count += 1
                                
                                # Если накопилось достаточно фреймов с речью, проверяем ключевое слово
                                if speech_frames_count >= 5:  # ~160ms речи (5 фреймов по 32ms)
                                    # Проверяем наличие ключевого слова с помощью OpenWakeWord
                                    if len(self.audio_buffer) >= self.sample_rate:  # Минимум 1 секунда аудио
                                        # Получаем аудио из буфера
                                        audio_data = np.array(list(self.audio_buffer))
                                        
                                        # Нормализуем аудио
                                        audio_data = audio_data / np.max(np.abs(audio_data) + 1e-10)
                                        
                                        # Получаем предсказание от OpenWakeWord
                                        prediction = self.oww_model.predict(audio_data)
                                        
                                        # Получаем вероятность для модели hey_jarvis
                                        score = prediction["hey_jarvis"]
                                        logger.debug(f"Вероятность hey_jarvis: {score:.4f}")
                                        
                                        # Логируем вероятность
                                        if score > 0.3:  # Логируем только если вероятность выше порога
                                            logger.info(f"Вероятность ключевого слова: {score:.4f}")
                                            print(f"{settings.Colors.SYSTEM}[WAKE] Вероятность ключевого слова: {score:.4f}{settings.Colors.END}")
                                        
                                        # Если вероятность выше порога, вызываем callback
                                        if score > 0.4:  # Настраиваемый порог
                                            logger.info(f"Обнаружено ключевое слово '{settings.WAKE_WORDS[0]}' с вероятностью {score:.4f}")
                                            print(f"{settings.Colors.SYSTEM}[WAKE] Обнаружено ключевое слово '{settings.WAKE_WORDS[0]}' с вероятностью {score:.4f}{settings.Colors.END}")
                                            
                                            # Сохраняем аудио во временный файл для дальнейшей обработки
                                            import soundfile as sf
                                            sf.write(settings.TEMP_WAV, audio_data, self.sample_rate)
                                            
                                            # Вызываем callback
                                            if self.callback:
                                                self.callback()
                                            
                                            # Сбрасываем счетчик
                                            speech_frames_count = 0
                            else:
                                # Сбрасываем счетчик, если речь не обнаружена
                                speech_frames_count = max(0, speech_frames_count - 1)
                            
                            # Очищаем VAD буфер для следующего чанка
                            self.vad_buffer.clear()
                    
                    self.audio_queue.task_done()
                except queue.Empty:
                    # Добавляем небольшую задержку при пустой очереди для снижения нагрузки на CPU
                    time.sleep(0.05)
                except Exception as e:
                    logger.error(f"Ошибка в цикле обработки аудио: {e}")
                    # Добавляем задержку при ошибке, чтобы не забивать лог
                    time.sleep(0.1)
        
        # Запускаем потоки записи и обработки аудио
        blocksize = int(self.sample_rate / 16000 * 512)  # Универсальная формула для разных частот
        
        try:
            # Создаем и запускаем аудио-стрим
            self.stream = sd.InputStream(callback=audio_callback, channels=settings.CHANNELS, 
                              samplerate=self.sample_rate, blocksize=blocksize)
            self.stream.start()
            
            # Запускаем поток обработки аудио
            self.audio_thread = threading.Thread(target=audio_processing)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Ждем, пока не будет установлен флаг остановки
            while self.is_running:
                time.sleep(0.2)  # Уменьшаем задержку для более быстрой реакции на остановку
            
            # Останавливаем поток обработки аудио
            stop_event.set()
            self.audio_queue.put(None)  # Отправляем poison pill
            self.audio_thread.join(timeout=2.0)
            
            # Останавливаем и закрываем стрим
            self.stream.stop()
            self.stream.close()
            
        except Exception as e:
            logger.error(f"Ошибка при запуске мониторинга: {e}")
            self.is_running = False