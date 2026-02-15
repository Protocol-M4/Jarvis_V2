import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import time
import queue
import threading
from collections import deque
from faster_whisper import WhisperModel
import os
import sys

# Магия импорта: позволяем Python видеть папку config, которая на уровень выше
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.logger import logger

class STTEngine:
    def __init__(self):
        # Инициализируем модель Whisper
        print(f"{settings.Colors.SYSTEM}[STT] Загрузка модели {settings.WHISPER_MODEL_SIZE}...{settings.Colors.END}")
        self.model = WhisperModel(
            settings.WHISPER_MODEL_SIZE, 
            device=settings.DEVICE, 
            compute_type=settings.COMPUTE_TYPE
        )
        
        # Инициализация Silero VAD
        print(f"{settings.Colors.SYSTEM}[VAD] Загрузка модели Silero VAD...{settings.Colors.END}")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              trust_repo=True)
        
        # Получаем только нужные функции из utils
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        
        # Переводим модель в режим оценки
        self.vad_model.eval()
        
        # Параметры VAD
        self.sample_rate = settings.SAMPLE_RATE
        self.silence_threshold = settings.VAD_SILENCE_THRESHOLD  # секунды тишины для завершения записи
        self.speech_threshold = settings.VAD_SPEECH_THRESHOLD    # порог вероятности речи
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 30))  # буфер на 30 секунд
        self.is_recording = False
        
    def record_to_file(self):
        """Запись с использованием VAD для определения начала и конца речи"""
        print(f"{settings.Colors.SYSTEM}[VAD] Слушаю...{settings.Colors.END}")
        
        # Очищаем буфер перед началом новой записи
        self.audio_buffer.clear()
        self.is_recording = False
        
        # Создаем очередь для обмена данными между потоками
        audio_queue = queue.Queue()
        stop_event = threading.Event()
        
        # Функция для записи аудио в отдельном потоке
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"{settings.Colors.ERROR}[VAD] Ошибка записи: {status}{settings.Colors.END}")
            # Преобразуем данные в формат, подходящий для VAD
            audio_chunk = indata.copy().flatten()
            audio_queue.put(audio_chunk)
        
        # Функция обработки VAD в отдельном потоке
        def vad_processing():
            silence_start = None
            consecutive_speech = 0
            required_size = int(self.sample_rate / 16000 * 512)  # Требуемый размер для VAD
            
            while not stop_event.is_set():
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                    
                    # Добавляем чанк в буфер для сохранения аудио
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Проверяем размер чанка для VAD
                    if len(audio_chunk) != required_size:
                        # Не логируем каждый случай, чтобы не засорять лог
                        audio_queue.task_done()
                        continue
                    
                    # Конвертируем numpy array в torch tensor
                    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
                    
                    # Прямой вызов модели
                    speech_prob = self.vad_model(audio_tensor, 16000).item()  # item() преобразует тензор в скаляр
                    
                    if speech_prob >= self.speech_threshold:
                        # Речь обнаружена
                        if not self.is_recording:
                            print(f"{settings.Colors.SYSTEM}[VAD] Речь обнаружена, запись...{settings.Colors.END}")
                            self.is_recording = True
                        
                        consecutive_speech += 1
                        silence_start = None
                    elif self.is_recording:
                        # Тишина после речи
                        if silence_start is None:
                            silence_start = time.time()
                        
                        # Проверяем, достаточно ли долго была тишина
                        if time.time() - silence_start > self.silence_threshold:
                            print(f"{settings.Colors.SYSTEM}[VAD] Речь окончена, обрабатываю...{settings.Colors.END}")
                            stop_event.set()  # Останавливаем запись
                    
                    audio_queue.task_done()
                except queue.Empty:
                    pass
        
        # Запускаем потоки записи и обработки VAD
        # Silero VAD требует ровно 512 сэмплов для частоты 16000 Гц
        blocksize = int(self.sample_rate / 16000 * 512)  # Универсальная формула для разных частот
        with sd.InputStream(callback=audio_callback, channels=settings.CHANNELS, 
                           samplerate=self.sample_rate, blocksize=blocksize):
            vad_thread = threading.Thread(target=vad_processing)
            vad_thread.start()
            
            # Ждем завершения записи или таймаута (максимум 30 секунд)
            stop_event.wait(timeout=30)
            stop_event.set()  # Гарантируем остановку потока VAD
            vad_thread.join()
        
        # Сохраняем записанное аудио в файл
        if len(self.audio_buffer) > 0:
            audio_data = np.array(list(self.audio_buffer))
            sf.write(settings.TEMP_WAV, audio_data, self.sample_rate)
            return True
        else:
            print(f"{settings.Colors.SYSTEM}[VAD] Речь не обнаружена{settings.Colors.END}")
            return False
    
    def wait_for_speech(self, timeout=7):
        """Ожидает начала речи в течение указанного времени"""
        print(f"{settings.Colors.SYSTEM}[SESSION] Жду продолжения ({timeout}с)...{settings.Colors.END}")
        
        start_time = time.time()
        speech_detected = False
        stop_event = threading.Event()
        
        # Функция для записи аудио в отдельном потоке
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"{settings.Colors.ERROR}[VAD] Ошибка записи: {status}{settings.Colors.END}")
            # Преобразуем данные в формат, подходящий для VAD
            audio_chunk = indata.copy().flatten()
            
            # Проверяем размер чанка
            required_size = int(self.sample_rate / 16000 * 512)
            if len(audio_chunk) != required_size:
                # Не логируем каждый случай, чтобы не засорять лог
                return
            
            # Конвертируем numpy array в torch tensor
            audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
            
            # Прямой вызов модели
            speech_prob = self.vad_model(audio_tensor, 16000).item()  # item() преобразует тензор в скаляр
            
            nonlocal speech_detected
            if speech_prob >= self.speech_threshold:
                speech_detected = True
                stop_event.set()
        
        # Запускаем поток записи с правильным размером блока
        blocksize = int(self.sample_rate / 16000 * 512)  # Универсальная формула для разных частот
        with sd.InputStream(callback=audio_callback, channels=settings.CHANNELS, 
                           samplerate=self.sample_rate, blocksize=blocksize):
            # Ждем либо обнаружения речи, либо истечения таймаута
            stop_event.wait(timeout=timeout)
        
        if speech_detected:
            print(f"{settings.Colors.SYSTEM}[VAD] Речь обнаружена, продолжаем сессию...{settings.Colors.END}")
        
        return speech_detected

    def transcribe_file(self):
        """Превращаем файл в текст"""
        try:
            segments, _ = self.model.transcribe(settings.TEMP_WAV, language="ru")
            text = "".join([s.text for s in segments])
            return text.strip()
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {e}")
            print(f"{settings.Colors.ERROR}[STT] Ошибка транскрипции: {e}{settings.Colors.END}")
            return ""
    
    def legacy_record_to_file(self):
        """Старый метод записи с фиксированным временем (запасной вариант)"""
        print(f"{settings.Colors.SYSTEM}>>> Слушаю (фиксированное время {settings.RECORD_SECONDS}с)...{settings.Colors.END}")
        # Используем параметры из нашего полного конфига
        audio_data = sd.rec(
            int(settings.RECORD_SECONDS * settings.SAMPLE_RATE), 
            samplerate=settings.SAMPLE_RATE, 
            channels=settings.CHANNELS
        )
        sd.wait()
        sf.write(settings.TEMP_WAV, audio_data, settings.SAMPLE_RATE)
        return True