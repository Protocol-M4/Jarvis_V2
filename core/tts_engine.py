import os
import sys
import time
import json
import wave
import pygame
import numpy as np
import soundfile as sf
from piper import PiperVoice
from piper.voice import PiperVoice
from core.logger import logger

# Магия импорта: позволяем Python видеть папку config, которая на уровень выше
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class TTSEngine:
    """
    Класс для синтеза речи с использованием Piper TTS.
    Piper - это локальный движок синтеза речи, который не требует подключения к интернету.
    """
    
    def __init__(self, voice_name="ru_RU-dmitri-medium"):
        """
        Инициализация голосового движка.
        
        Args:
            voice_name: Имя голосовой модели (без расширения)
        """
        logger.info(f"Инициализация голосового движка Piper TTS с голосом {voice_name}...")
        
        # Пути к файлам
        self.models_dir = os.path.join("models", "piper")
        self.temp_file_piper = "temp_output_piper.wav"
        self.temp_file_edge = "temp_output_edge.mp3"
        self.temp_file_converted = "temp_output_converted.wav"  # Для конвертации MP3 в WAV
        self.temp_file = self.temp_file_piper  # По умолчанию используем Piper
        
        # Проверяем наличие директории с моделями
        if not os.path.exists(self.models_dir):
            os.makedirs(models_dir, exist_ok=True)
            logger.warning(f"Директория {self.models_dir} не существовала и была создана")
        
        # Пути к файлам модели
        self.model_path = os.path.join(self.models_dir, f"{voice_name}.onnx")
        self.config_path = os.path.join(self.models_dir, f"{voice_name}.onnx.json")
        
        # Проверяем наличие и валидность файлов модели
        model_valid = False
        if os.path.exists(self.model_path) and os.path.exists(self.config_path):
            # Проверяем размер файлов
            model_size = os.path.getsize(self.model_path)
            config_size = os.path.getsize(self.config_path)
            
            # Модель должна быть не менее 1 МБ, конфиг - не менее 100 байт
            if model_size > 1024 * 1024 and config_size > 100:
                model_valid = True
                
                # Проверяем валидность JSON-файла конфигурации
                try:
                    with open(self.config_path, 'r') as f:
                        json.load(f)  # Пробуем загрузить JSON
                except json.JSONDecodeError:
                    model_valid = False
                    logger.error(f"Файл конфигурации {self.config_path} содержит невалидный JSON")
            else:
                logger.warning(f"Файлы модели {voice_name} имеют подозрительно малый размер: модель {model_size/1024:.1f} КБ, конфиг {config_size} байт")
        
        if not model_valid:
            # Если файлов нет или они невалидны, используем Edge TTS как запасной вариант
            logger.warning(f"Файлы модели {voice_name} не найдены или повреждены. Используем Edge TTS как запасной вариант.")
            self.use_edge_tts = True
            self.edge_voice = "ru-RU-DmitryNeural"
            
            # Импортируем edge_tts только если он нужен
            import edge_tts
            self.edge_tts = edge_tts
        else:
            # Если файлы есть и они валидны, используем Piper
            self.use_edge_tts = False
            try:
                # Инициализируем Piper с явным указанием use_cuda=False для Mac M4
                logger.info(f"Загрузка модели Piper из {self.model_path}...")
                self.piper = PiperVoice.load(self.model_path, config_path=self.config_path, use_cuda=False)
                
                # Выводим информацию о типе объекта для отладки
                logger.info(f"Тип объекта Piper: {type(self.piper)}")
                if hasattr(self.piper, 'voice'):
                    logger.info(f"Тип объекта self.piper.voice: {type(self.piper.voice)}")
                
                logger.info("Модель Piper успешно загружена")
            except Exception as e:
                # Если не удалось загрузить Piper, используем Edge TTS как запасной вариант
                logger.error(f"Ошибка при загрузке модели Piper: {e}")
                logger.warning("Используем Edge TTS как запасной вариант")
                self.use_edge_tts = True
                self.edge_voice = "ru-RU-DmitryNeural"
                
                # Импортируем edge_tts только если он нужен
                import edge_tts
                self.edge_tts = edge_tts
        
        # Инициализируем микшер pygame, если он еще не запущен
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        logger.info("Голосовой движок инициализирован")
    
    async def _generate_audio_with_edge_tts(self, text, retries=3):
        """
        Генерирует аудио с использованием Edge TTS с защитой от сбоев интернета.
        
        Args:
            text: Текст для озвучивания
            retries: Количество попыток при сбоях
            
        Returns:
            bool: True, если генерация успешна, иначе False
        """
        # Импортируем asyncio здесь, чтобы избежать циклических импортов
        import asyncio
        
        # Используем MP3 для Edge TTS
        self.temp_file = self.temp_file_edge
        
        for attempt in range(retries):
            try:
                communicate = self.edge_tts.Communicate(text, self.edge_voice)
                await communicate.save(self.temp_file)
                return True  # Успех
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Попытка {attempt + 1} не удалась. Пробую снова через 1 сек...")
                    print(f"[TTS] Попытка {attempt + 1} не удалась. Пробую снова через 1 сек...")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Ошибка подключения к серверу голоса: {e}")
                    print(f"[TTS] Ошибка подключения к серверу голоса: {e}")
                    return False
    
    def _generate_audio_with_piper(self, text):
        """
        Генерирует аудио с использованием Piper TTS.
        
        Args:
            text: Текст для озвучивания
            
        Returns:
            bool: True, если генерация успешна, иначе False
        """
        # Используем WAV для Piper
        self.temp_file = self.temp_file_piper
        
        try:
            # Открываем WAV файл для записи
            with wave.open(self.temp_file, "wb") as wav_file:
                # Настраиваем параметры WAV файла
                wav_file.setnchannels(1)  # Моно
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.piper.config.sample_rate)  # Частота дискретизации из конфига
                
                # Счетчик для отслеживания размера данных
                total_bytes = 0
                
                # Итерируемся по генератору и записываем аудио-фрагменты напрямую
                for chunk in self.piper.synthesize(text):
                    # Логируем тип объекта для первого чанка
                    if total_bytes == 0:
                        logger.info(f"Тип аудио-чанка: {type(chunk)}")
                        logger.info(f"Атрибуты аудио-чанка: {dir(chunk)}")
                    
                    # Используем audio_int16_bytes напрямую
                    if hasattr(chunk, 'audio_int16_bytes'):
                        bytes_data = chunk.audio_int16_bytes
                        if bytes_data:
                            wav_file.writeframes(bytes_data)
                            total_bytes += len(bytes_data)
                        else:
                            logger.warning("Piper Engine Warning: Generator returned empty frames")
                            print("[TTS] Piper Engine Warning: Generator returned empty frames")
            
            # Проверяем размер сгенерированного файла
            file_size = os.path.getsize(self.temp_file)
            logger.info(f"Аудио сгенерировано, размер файла: {file_size} байт")
            
            if file_size > 44:  # 44 байта - это размер заголовка WAV без данных
                logger.info(f"Аудио успешно сгенерировано и сохранено в {self.temp_file}")
                return True
            else:
                logger.warning(f"Файл {self.temp_file} содержит только заголовок WAV без аудио данных")
                return False
            
        except Exception as e:
            logger.error(f"Ошибка при генерации аудио с Piper: {e}")
            print(f"[TTS] Ошибка при генерации аудио с Piper: {e}")
            return False
    
    def _convert_mp3_to_wav(self, mp3_file, wav_file):
        """
        Конвертирует MP3 в WAV для корректного воспроизведения.
        
        Args:
            mp3_file: Путь к MP3 файлу
            wav_file: Путь для сохранения WAV файла
            
        Returns:
            bool: True, если конвертация успешна, иначе False
        """
        try:
            # Используем pygame для конвертации
            import pygame.mixer
            
            # Загружаем MP3
            sound = pygame.mixer.Sound(mp3_file)
            
            # Получаем аудиоданные
            array_sample = pygame.sndarray.array(sound)
            
            # Сохраняем как WAV
            import scipy.io.wavfile
            scipy.io.wavfile.write(wav_file, pygame.mixer.get_init()[0], array_sample)
            
            return True
        except Exception as e:
            logger.error(f"Ошибка при конвертации MP3 в WAV: {e}")
            print(f"[TTS] Ошибка при конвертации MP3 в WAV: {e}")
            
            # Альтернативный метод конвертации
            try:
                from pydub import AudioSegment
                sound = AudioSegment.from_mp3(mp3_file)
                sound.export(wav_file, format="wav")
                return True
            except Exception as e2:
                logger.error(f"Альтернативная конвертация также не удалась: {e2}")
                print(f"[TTS] Альтернативная конвертация также не удалась: {e2}")
                return False
    
    def say(self, text):
        """
        Основной метод для озвучки текста.
        
        Args:
            text: Текст для озвучивания
        """
        if not text:
            return
        
        # Чистим текст от артефактов Markdown (ИИ любит их ставить)
        clean_text = text.replace('*', '').replace('_', '').replace('#', '').strip()
        
        try:
            # Генерируем аудио
            if self.use_edge_tts:
                # Если используем Edge TTS, генерируем асинхронно
                import asyncio
                success = asyncio.run(self._generate_audio_with_edge_tts(clean_text))
            else:
                # Если используем Piper, генерируем синхронно
                success = self._generate_audio_with_piper(clean_text)
            
            if not success:
                logger.warning("Не удалось сгенерировать аудио. Пропускаем воспроизведение.")
                print("[TTS] Не удалось сгенерировать аудио. Пропускаем воспроизведение.")
                return  # Если генерация не удалась, просто выходим без звука
            
            # Проверяем, существует ли файл
            if not os.path.exists(self.temp_file):
                logger.error(f"Файл {self.temp_file} не существует после генерации аудио")
                print(f"[TTS] Файл {self.temp_file} не существует после генерации аудио")
                return
            
            # Проверяем размер файла
            file_size = os.path.getsize(self.temp_file)
            if file_size < 100:  # Если файл слишком маленький, считаем его некорректным
                logger.error(f"Файл {self.temp_file} имеет подозрительно малый размер ({file_size} байт)")
                print(f"[TTS] Файл {self.temp_file} имеет подозрительно малый размер ({file_size} байт)")
                return
            
            # Если используем Edge TTS (MP3), конвертируем в WAV для надежности
            play_file = self.temp_file
            if self.use_edge_tts and self.temp_file.endswith('.mp3'):
                logger.info("Конвертация MP3 в WAV для надежного воспроизведения...")
                if self._convert_mp3_to_wav(self.temp_file, self.temp_file_converted):
                    play_file = self.temp_file_converted
                    logger.info(f"Файл успешно конвертирован в {play_file}")
                else:
                    logger.warning("Конвертация не удалась, пробуем воспроизвести MP3 напрямую")
            
            try:
                # Воспроизведение через pygame
                pygame.mixer.music.load(play_file)
                pygame.mixer.music.play()
                
                # Ждем, пока файл доиграет до конца
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Выгружаем файл
                pygame.mixer.music.unload()
            except Exception as e:
                logger.error(f"Ошибка при воспроизведении аудио: {e}")
                print(f"[TTS] Ошибка при воспроизведении аудио: {e}")
                
                # Если не удалось воспроизвести, пробуем альтернативный метод
                try:
                    logger.info("Пробуем альтернативный метод воспроизведения...")
                    import simpleaudio as sa
                    wave_obj = sa.WaveObject.from_wave_file(play_file)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                    logger.info("Альтернативное воспроизведение успешно")
                except Exception as e2:
                    logger.error(f"Альтернативное воспроизведение также не удалось: {e2}")
                    print(f"[TTS] Альтернативное воспроизведение также не удалось: {e2}")
            
            # Удаляем временные файлы
            try:
                for temp_file in [self.temp_file, self.temp_file_converted]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except Exception as e:
                logger.error(f"Ошибка при удалении временных файлов: {e}")
                print(f"[TTS] Ошибка при удалении временных файлов: {e}")
        
        except Exception as e:
            logger.error(f"Критическая ошибка в модуле озвучки: {e}")
            print(f"[TTS Critical Error] Проблема в модуле озвучки: {e}")

    def __del__(self):
        """Закрываем микшер при выходе, если нужно"""
        try:
            # Удаляем временные файлы, если они существуют
            for temp_file in [self.temp_file_piper, self.temp_file_edge, self.temp_file_converted]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Закрываем микшер
            pygame.mixer.quit()
            
            logger.info("Голосовой движок освобожден")
        except Exception as e:
            logger.error(f"Ошибка при освобождении ресурсов голосового движка: {e}")
