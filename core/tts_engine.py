import edge_tts
import pygame
import asyncio
import os
import time

class TTSEngine:
    def __init__(self, voice="ru-RU-DmitryNeural"):
        """
        Инициализация голосового движка.
        voice: выбранный голос (DmitryNeural — отличный мужской бас)
        """
        self.voice = voice
        self.temp_file = "temp_output.mp3"
        
        # Инициализируем микшер pygame, если он еще не запущен
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    async def _generate_audio_with_retry(self, text, retries=3):
        """Генерирует аудио с защитой от сбоев интернета"""
        for attempt in range(retries):
            try:
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(self.temp_file)
                return True  # Успех
            except Exception as e:
                if attempt < retries - 1:
                    print(f"[TTS] Попытка {attempt + 1} не удалась. Пробую снова через 1 сек...")
                    await asyncio.sleep(1)
                else:
                    print(f"[TTS] Ошибка подключения к серверу голоса: {e}")
                    return False

    def say(self, text):
        """Основной метод для озвучки текста"""
        if not text:
            return

        # 1. Чистим текст от артефактов Markdown (ИИ любит их ставить)
        clean_text = text.replace('*', '').replace('_', '').replace('#', '').strip()
        
        try:
            # 2. Асинхронно генерируем файл
            success = asyncio.run(self._generate_audio_with_retry(clean_text))
            
            if not success:
                return # Если интернет так и не поднялся, просто выходим без звука

            # 3. Воспроизведение через pygame
            pygame.mixer.music.load(self.temp_file)
            pygame.mixer.music.play()
            
            # Ждем, пока файл доиграет до конца
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # 4. Выгружаем файл и удаляем его (гигиена проекта)
            pygame.mixer.music.unload()
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
                
        except Exception as e:
            print(f"[TTS Critical Error] Проблема в модуле озвучки: {e}")

    def __del__(self):
        """Закрываем микшер при выходе, если нужно"""
        try:
            pygame.mixer.quit()
        except:
            pass
