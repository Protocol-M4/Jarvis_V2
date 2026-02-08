import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import os
import sys

# Магия импорта: позволяем Python видеть папку config, которая на уровень выше
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class STTEngine:
    def __init__(self):
        # Инициализируем модель один раз при старте!
        print(f"{settings.Colors.SYSTEM}[STT] Загрузка модели {settings.WHISPER_MODEL_SIZE}...{settings.Colors.END}")
        self.model = WhisperModel(
            settings.WHISPER_MODEL_SIZE, 
            device=settings.DEVICE, 
            compute_type=settings.COMPUTE_TYPE
        )

    def record_to_file(self):
        print(f"{settings.Colors.SYSTEM}>>> Слушаю...{settings.Colors.END}")
        # Используем параметры из нашего полного конфига
        audio_data = sd.rec(
            int(settings.RECORD_SECONDS * settings.SAMPLE_RATE), 
            samplerate=settings.SAMPLE_RATE, 
            channels=settings.CHANNELS
        )
        sd.wait()
        sf.write(settings.TEMP_WAV, audio_data, settings.SAMPLE_RATE)

    def transcribe_file(self):
        # Превращаем файл в текст
        segments, _ = self.model.transcribe(settings.TEMP_WAV, language="ru")
        text = "".join([s.text for s in segments])
        return text.strip()
