import os
import sys
import time
import queue
import threading
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
from collections import deque
from faster_whisper import WhisperModel

# Магия импорта: позволяем Python видеть папку config, которая на уровень выше
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.logger import logger

class WakeWordDetector:
    """
    Класс для обнаружения ключевого слова (wake word) с использованием
    каскадной системы: Silero VAD -> Whisper tiny
    """
    
    def __init__(self):
        """Инициализация детектора ключевого слова"""
        logger.info("Инициализация системы обнаружения ключевого слова...")
        
        # Инициализация Silero VAD
        logger.info(f"Загрузка модели Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              trust_repo=True)
        
        # Получаем только нужные функции из utils
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        
        # Переводим модель в режим оценки
        self.vad_model.eval()
        
        # Инициализация Whisper для проверки ключевого слова
        logger.info(f"Загрузка модели Whisper {settings.WAKE_WORD_MODEL_SIZE} для проверки ключевого слова...")
        self.whisper_model = WhisperModel(
            settings.WAKE_WORD_MODEL_SIZE, 
            device=settings.DEVICE, 
            compute_type=settings.COMPUTE_TYPE
        )
        
        # Параметры
        self.sample_rate = settings.SAMPLE_RATE
        self.speech_threshold = settings.WAKE_WORD_THRESHOLD
        self.buffer_size = int(self.sample_rate * settings.WAKE_WORD_BUFFER_SIZE)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Флаги состояния
        self.is_running = False
        self.speech_detected = False
        
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
        
        # Очищаем буфер аудио
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer.clear()
        
        logger.info("Мониторинг ключевого слова остановлен")
    
    def _monitoring_loop(self):
        """Основной цикл мониторинга аудио"""
        # Очищаем буфер перед началом
        self.audio_buffer.clear()
        
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
            # Преобразуем данные в формат, подходящий для VAD
            audio_chunk = indata.copy().flatten()
            if self.is_running:  # Проверяем флаг перед добавлением в очередь
                self.audio_queue.put(audio_chunk)
        
        # Функция обработки VAD в отдельном потоке
        def vad_processing():
            required_size = int(self.sample_rate / 16000 * 512)  # Требуемый размер для VAD
            speech_frames_count = 0
            
            while not stop_event.is_set() and self.is_running:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    # Проверяем на poison pill
                    if audio_chunk is None:
                        logger.info("Получен сигнал завершения (poison pill)")
                        break
                    
                    # Добавляем чанк в буфер для сохранения аудио
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Проверяем размер чанка для VAD
                    if len(audio_chunk) != required_size:
                        self.audio_queue.task_done()
                        continue
                    
                    # Конвертируем numpy array в torch tensor
                    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
                    
                    # Прямой вызов модели
                    speech_prob = self.vad_model(audio_tensor, 16000).item()
                    
                    if speech_prob >= self.speech_threshold:
                        # Речь обнаружена, увеличиваем счетчик
                        speech_frames_count += 1
                        
                        # Если накопилось достаточно фреймов с речью, проверяем ключевое слово
                        if speech_frames_count >= 5:  # ~160ms речи (5 фреймов по 32ms)
                            # Получаем метки речи с минимальной длительностью 300 мс
                            audio_data = np.array(list(self.audio_buffer))
                            audio_tensor_full = torch.tensor(audio_data, dtype=torch.float32)
                            speech_timestamps = self.get_speech_timestamps(
                                audio_tensor_full, 
                                self.vad_model,
                                threshold=self.speech_threshold,
                                min_speech_duration_ms=300,
                                sampling_rate=self.sample_rate
                            )
                            
                            # Проверяем наличие речи достаточной длительности
                            if speech_timestamps:
                                # Проверяем ключевое слово
                                if self._check_wake_word():
                                    # Ключевое слово обнаружено, вызываем callback
                                    if self.callback:
                                        self.callback()
                                    # Сбрасываем счетчик
                                    speech_frames_count = 0
                    else:
                        # Сбрасываем счетчик, если речь не обнаружена
                        speech_frames_count = max(0, speech_frames_count - 1)
                    
                    self.audio_queue.task_done()
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error(f"Ошибка в цикле обработки VAD: {e}")
        
        # Запускаем потоки записи и обработки VAD
        blocksize = int(self.sample_rate / 16000 * 512)  # Универсальная формула для разных частот
        
        try:
            # Создаем и запускаем аудио-стрим
            self.stream = sd.InputStream(callback=audio_callback, channels=settings.CHANNELS, 
                              samplerate=self.sample_rate, blocksize=blocksize)
            self.stream.start()
            
            # Запускаем поток обработки VAD
            self.vad_thread = threading.Thread(target=vad_processing)
            self.vad_thread.daemon = True
            self.vad_thread.start()
            
            # Ждем, пока не будет установлен флаг остановки
            while self.is_running:
                time.sleep(0.5)  # Увеличиваем задержку для снижения нагрузки на процессор
            
            # Останавливаем поток VAD
            stop_event.set()
            self.audio_queue.put(None)  # Отправляем poison pill
            self.vad_thread.join(timeout=2.0)
            
            # Останавливаем и закрываем стрим
            self.stream.stop()
            self.stream.close()
            
        except Exception as e:
            logger.error(f"Ошибка при запуске мониторинга: {e}")
            self.is_running = False
    
    def _check_wake_word(self):
        """
        Проверяет наличие ключевого слова в буфере аудио и обрезает буфер
        
        Returns:
            bool: True, если ключевое слово обнаружено, иначе False
        """
        try:
            # Отладочный вывод в начале метода
            logger.info("Проверка на наличие ключевого слова в аудио-буфере...")
            print(f"{settings.Colors.SYSTEM}[WAKE] Проверка на наличие ключевого слова '{settings.WAKE_WORDS[0]}' в аудио-буфере...{settings.Colors.END}")
            
            # Сохраняем буфер во временный файл
            temp_file = settings.TEMP_WAV
            audio_data = np.array(list(self.audio_buffer))
            
            # Логируем длину буфера до обрезки
            buffer_length = len(audio_data) / self.sample_rate
            logger.info(f"Буфер до обрезки: {buffer_length:.2f} секунд")
            print(f"{settings.Colors.SYSTEM}[WAKE] Буфер до обрезки: {buffer_length:.2f} секунд{settings.Colors.END}")
            
            # Проверяем, достаточно ли длинный буфер для транскрипции
            if buffer_length < 2.0:  # Минимум 2 секунды аудио
                logger.info("Буфер слишком короткий для транскрипции, пропускаем")
                print(f"{settings.Colors.SYSTEM}[WAKE] Буфер слишком короткий для транскрипции, пропускаем{settings.Colors.END}")
                return False
            
            # Проверяем наличие речи в буфере с помощью VAD
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                threshold=self.speech_threshold,
                min_speech_duration_ms=300,
                sampling_rate=self.sample_rate
            )
            
            # Если речь не обнаружена, пропускаем транскрипцию
            if not speech_timestamps:
                logger.info("Речь не обнаружена в буфере, пропускаем транскрипцию")
                print(f"{settings.Colors.SYSTEM}[WAKE] Речь не обнаружена в буфере, пропускаем транскрипцию{settings.Colors.END}")
                return False
            
            # Логируем информацию о найденной речи
            speech_duration_ms = sum([t['end'] - t['start'] for t in speech_timestamps])
            logger.info(f"Обнаружена речь в буфере, длительность: {speech_duration_ms/self.sample_rate*1000:.0f} мс")
            print(f"{settings.Colors.SYSTEM}[WAKE] Обнаружена речь в буфере, длительность: {speech_duration_ms/self.sample_rate*1000:.0f} мс{settings.Colors.END}")
            
            # Сохраняем буфер во временный файл
            sf.write(temp_file, audio_data, self.sample_rate)
            
            try:
                # Транскрибируем аудио с помощью Whisper
                segments, _ = self.whisper_model.transcribe(temp_file, language="ru")
                
                # Логируем все распознанные сегменты
                for segment in segments:
                    print(f"{settings.Colors.SYSTEM}[WAKE] Распознано: {segment.text}{settings.Colors.END}")
                
                # Ищем сегмент с ключевым словом
                wake_word_detected = False
                wake_word_segment = None
                
                for segment in segments:
                    if any(word in segment.text.lower() for word in settings.WAKE_WORDS):
                        wake_word_detected = True
                        wake_word_segment = segment
                        # Определяем, какое именно ключевое слово было обнаружено
                        detected_word = next((word for word in settings.WAKE_WORDS if word in segment.text.lower()), settings.WAKE_WORDS[0])
                        break
                
                if wake_word_detected and wake_word_segment:
                    # Определяем, какое именно ключевое слово было обнаружено
                    detected_word = next((word for word in settings.WAKE_WORDS if word in wake_word_segment.text.lower()), settings.WAKE_WORDS[0])
                    logger.info(f"Обнаружено ключевое слово: '{detected_word}' в тексте: '{wake_word_segment.text}'")
                    
                    # Обрезаем аудио-буфер, начиная с позиции ключевого слова
                    # Whisper возвращает время в секундах, переводим в сэмплы
                    start_sample = int(wake_word_segment.start * self.sample_rate)
                    
                    # Создаем новый буфер, начиная с позиции ключевого слова
                    # Но отступаем немного назад (0.2 секунды), чтобы гарантированно захватить слово
                    safety_margin = int(0.2 * self.sample_rate)
                    start_sample = max(0, start_sample - safety_margin)
                    
                    # Обрезаем буфер
                    trimmed_audio = audio_data[start_sample:]
                    
                    # Логируем длину буфера после обрезки
                    buffer_length_after = len(trimmed_audio) / self.sample_rate
                    logger.info(f"Буфер после обрезки: {buffer_length_after:.2f} секунд (сокращение на {buffer_length - buffer_length_after:.2f} секунд)")
                    print(f"{settings.Colors.SYSTEM}[WAKE] Буфер после обрезки: {buffer_length_after:.2f} секунд (сокращение на {buffer_length - buffer_length_after:.2f} секунд){settings.Colors.END}")
                    
                    # Сохраняем обрезанный буфер
                    sf.write(settings.TEMP_WAV, trimmed_audio, self.sample_rate)
                    
                    # Обновляем аудио-буфер
                    self.audio_buffer.clear()
                    self.audio_buffer.extend(trimmed_audio)
                    
                    logger.info(f"Аудио-буфер обрезан, начиная с позиции ключевого слова (с запасом {safety_margin} сэмплов)")
                elif wake_word_detected:
                    # Если ключевое слово обнаружено, но сегмент не найден (редкий случай)
                    text = "".join([s.text for s in segments]).lower()
                    logger.info(f"Обнаружено ключевое слово в тексте: '{text}', но не удалось определить позицию")
                
                return wake_word_detected
                
            except RuntimeWarning as w:
                logger.error(f"RuntimeWarning при транскрипции: {w}")
                print(f"{settings.Colors.ERROR}[WAKE] RuntimeWarning при транскрипции: {w}{settings.Colors.END}")
                # Очищаем буфер при ошибке
                self.audio_buffer.clear()
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при проверке ключевого слова: {e}")
            print(f"{settings.Colors.ERROR}[WAKE] Ошибка при проверке ключевого слова: {e}{settings.Colors.END}")
            # Очищаем буфер при ошибке
            self.audio_buffer.clear()
            return False
