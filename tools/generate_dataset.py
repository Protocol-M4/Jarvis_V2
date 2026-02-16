#!/usr/bin/env python3
"""
Скрипт для генерации датасета для обучения модели OpenWakeWord.
Создает аудиофайлы с произношением ключевого слова "Джарвис"
и негативные примеры без ключевого слова.
Оптимизирован для создания универсального русского верификатора.
Использует gTTS для прямой записи аудио в файл.
"""

import os
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
import argparse
import sys
import torch
import torchaudio
import torchaudio.transforms as T
import subprocess
import tempfile
from gtts import gTTS
import librosa
import librosa.effects

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.tts_engine import TTSEngine

# Параметры генерации
SAMPLE_RATE = 16000  # OpenWakeWord требует 16 кГц
CHANNELS = 1  # Моно

# Варианты произношения "Джарвис" (только русские варианты)
VARIANTS = [
    # Базовые варианты
    "Джарвис", "Джа́рвис", "Джарви́с",
    
    # Только одиночное слово с разными ударениями
    "Джа́рвис", "Джарви́с", "Джа́рви́с",
    
    # Вариации произношения
    "Джарвис", "Джэрвис", "Жарвис", "Ярвис",
    
    # Простые обращения (без дополнительных команд)
    "Джарвис", "Джарвис!", "Джарвис?", "Джарвис."
]

# Варианты скорости произношения для TTS
SPEED_OPTIONS = [False]  # Используем только нормальную скорость, затем ускоряем программно

# Расширенный список русских слов и фраз (не содержащих "Джарвис")
NEGATIVE_PHRASES = [
    "Привет", "Как дела", "Что нового", "Погода хорошая", 
    "Включи музыку", "Который час", "Спасибо", "До свидания",
    "Доброе утро", "Добрый день", "Добрый вечер", "Спокойной ночи",
    "Помоги мне", "Расскажи новости", "Включи свет", "Выключи свет",
    "Поставь будильник", "Напомни мне", "Запиши в календарь", "Позвони",
    "Отправь сообщение", "Найди информацию", "Сколько времени", "Какая погода",
    "Включи телевизор", "Выключи телевизор", "Сделай громче", "Сделай тише",
    "Переключи канал", "Открой приложение", "Закрой приложение", "Запусти программу",
    "Останови программу", "Перезагрузи компьютер", "Выключи компьютер", "Включи компьютер",
    "Сохрани файл", "Удали файл", "Создай папку", "Удали папку",
    "Скопируй файл", "Переименуй файл", "Переместись в папку", "Вернись назад",
    "Открой браузер", "Закрой браузер", "Найди в интернете", "Загрузи файл",
    "Алиса", "Сири", "Окей Гугл", "Алекса", "Кортана"
]

# Hard negatives - созвучные с "Джарвис" слова
HARD_NEGATIVES = [
    "Барвис", "Дарвис", "Рис", "Сервис", "Марвис", 
    "Джаз", "Сервиз", "Нарцисс", "Карниз", "Дервиш",
    "Сервер", "Джаз-бэнд", "Сервисный центр", "Нарисуй", "Дарвинизм",
    "Барристер", "Сервировка", "Карвинг", "Дарвин", "Харрис",
    "Борис", "Денис", "Ларисса", "Кларисса", "Маркиз",
    "Барбарис", "Дарвинист", "Сюрприз", "Каприз", "Маркиза"
]

# Добавляем hard negatives в основной список
NEGATIVE_PHRASES.extend(HARD_NEGATIVES)

def add_noise(audio, noise_level=0.01):
    """Добавляет шум к аудио"""
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise

def change_speed(audio, speed_factor, sr=16000):
    """
    Изменяет скорость аудио без изменения высоты тона.
    
    Args:
        audio: Аудиоданные
        speed_factor: Коэффициент изменения скорости (>1.0 - ускорение, <1.0 - замедление)
        sr: Частота дискретизации
    
    Returns:
        Аудиоданные с измененной скоростью
    """
    # Используем librosa для изменения скорости без изменения высоты тона
    try:
        # В librosa.effects.time_stretch параметр rate означает:
        # rate < 1.0: замедление (растягивание аудио)
        # rate > 1.0: ускорение (сжатие аудио)
        # Поэтому для ускорения нам нужно использовать rate=speed_factor
        y_stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed_factor)
        return y_stretched
    except Exception as e:
        print(f"Ошибка при изменении скорости с librosa: {e}")
        # Запасной вариант - простой ресемплинг
        indices = np.round(np.arange(0, len(audio), 1.0/speed_factor)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]

def generate_sine_wave(freq, duration, sample_rate=SAMPLE_RATE):
    """Генерирует синусоидальную волну заданной частоты и длительности"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave

def generate_tts_speech(text, output_file, slow=False):
    """
    Генерирует речь с использованием gTTS и сохраняет напрямую в файл.
    Применяет аугментацию для ускорения речи.
    
    Args:
        text: Текст для озвучивания
        output_file: Путь для сохранения аудиофайла
        slow: Флаг медленного произношения
    
    Returns:
        bool: True, если генерация успешна, иначе False
    """
    try:
        # Создаем объект gTTS
        tts = gTTS(text=text, lang='ru', slow=slow)
        
        # Сохраняем аудио напрямую в файл
        tts.save(output_file)
        
        # Проверяем размер файла
        file_size = os.path.getsize(output_file)
        
        if file_size > 1000:  # Файл должен быть больше 1 КБ
            # Загружаем аудио для возможной модификации
            audio, sr = sf.read(output_file)
            
            # Применяем ускорение речи (аугментация) с сохранением высоты тона
            # Ускоряем речь на 10-30% для более естественного звучания
            speed_factor = random.uniform(1.1, 1.3)  # 1.1 = ускорение на 10%, 1.3 = ускорение на 30%
            audio = change_speed(audio, speed_factor, sr)
            
            # Добавляем небольшой шум для разнообразия
            noise_level = random.uniform(0.001, 0.005)
            audio = add_noise(audio, noise_level)
            
            # Сохраняем модифицированное аудио
            sf.write(output_file, audio, sr)
            
            return True
        else:
            print(f"Ошибка: файл {output_file} слишком маленький ({file_size} байт)")
            # Запасной вариант - используем синтетическую речь
            duration = random.uniform(1.0, 2.0)
            return generate_synthetic_speech(text, output_file, duration)
    except Exception as e:
        print(f"Ошибка при генерации TTS: {e}")
        # Запасной вариант - используем синтетическую речь
        print(f"Используем синтетическую речь")
        duration = random.uniform(1.0, 2.0)
        return generate_synthetic_speech(text, output_file, duration)

def generate_synthetic_speech(text, output_file, duration=1.0):
    """
    Генерирует синтетическую 'речь' на основе текста.
    Это запасной вариант, если TTS не работает.
    """
    # Создаем базовую частоту на основе хеша текста
    base_freq = hash(text) % 400 + 200  # Частота от 200 до 600 Гц
    
    # Генерируем аудио
    audio = np.zeros(int(SAMPLE_RATE * duration))
    words = text.split()
    
    # Защита от деления на ноль, если текст пустой
    if not words:
        words = [""]
    
    word_duration = duration / len(words)
    
    for i, word in enumerate(words):
        # Разная частота для каждого слова
        freq = base_freq + (hash(word) % 200)
        start_idx = int(i * word_duration * SAMPLE_RATE)
        end_idx = int((i + 1) * word_duration * SAMPLE_RATE)
        end_idx = min(end_idx, len(audio))
        
        # Генерируем волну для слова
        word_audio = generate_sine_wave(freq, word_duration)
        
        # Убеждаемся, что длина word_audio точно соответствует нужному размеру
        segment_length = end_idx - start_idx
        if len(word_audio) > segment_length:
            word_audio = word_audio[:segment_length]
        elif len(word_audio) < segment_length:
            # Если word_audio короче, дополняем его нулями
            padding = np.zeros(segment_length - len(word_audio))
            word_audio = np.concatenate([word_audio, padding])
        
        # Добавляем в общий аудиофайл
        audio[start_idx:end_idx] = word_audio
    
    # Добавляем шум
    audio = add_noise(audio, 0.01)
    
    # Сохраняем в файл
    sf.write(output_file, audio, SAMPLE_RATE)
    
    return output_file

def generate_dataset(output_dir, num_positive=10000, num_negative=3000):
    """Генерирует датасет для обучения модели OpenWakeWord"""
    # Создаем директории для датасета
    os.makedirs(output_dir, exist_ok=True)
    positive_dir = os.path.join(output_dir, "positive")
    negative_dir = os.path.join(output_dir, "negative")
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    
    # Генерация позитивных примеров с использованием gTTS
    print(f"Генерация {num_positive} позитивных примеров с использованием gTTS...")
    for i in tqdm(range(num_positive)):
        # Выбираем вариант произношения
        variant = random.choice(VARIANTS)
        
        # Определяем скорость произношения (нормальная или медленная)
        slow = random.choice(SPEED_OPTIONS)
        
        # Генерируем аудио с использованием gTTS
        output_file = os.path.join(positive_dir, f"jarvis_{i:04d}.wav")
        
        # Пробуем использовать gTTS, если не получается - используем синтетическую речь
        tts_success = generate_tts_speech(variant, output_file, slow)
        
        if not tts_success:
            # Если TTS не сработал, используем запасной вариант
            print(f"Используем запасной вариант для примера {i}")
            duration = random.uniform(1.0, 2.0)
            generate_synthetic_speech(variant, output_file, duration)
            
            # Добавляем случайные модификации к аудио
            try:
                audio, sr = sf.read(output_file)
                
                # Добавляем шум с вероятностью 50%
                if random.random() > 0.5:
                    noise_level = random.uniform(0.005, 0.02)
                    audio = add_noise(audio, noise_level)
                
                # Изменяем скорость с вероятностью 30%
                if random.random() > 0.7:
                    speed_factor = random.uniform(0.9, 1.1)
                    audio = change_speed(audio, speed_factor)
                
                sf.write(output_file, audio, sr)
            except Exception as e:
                print(f"Ошибка при обработке файла {output_file}: {e}")
    
    # Генерация негативных примеров
    print(f"Генерация {num_negative} негативных примеров...")
    for i in tqdm(range(num_negative)):
        # С вероятностью 30% выбираем hard negative
        if random.random() < 0.3 and HARD_NEGATIVES:
            phrase = random.choice(HARD_NEGATIVES)
        else:
            phrase = random.choice(NEGATIVE_PHRASES)
        
        # Генерируем аудио
        output_file = os.path.join(negative_dir, f"not_jarvis_{i:04d}.wav")
        duration = random.uniform(1.0, 3.0)  # Случайная длительность от 1 до 3 секунд
        generate_synthetic_speech(phrase, output_file, duration)
        
        # Добавляем случайные модификации к аудио
        try:
            audio, sr = sf.read(output_file)
            
            # Добавляем шум с вероятностью 70%
            if random.random() > 0.3:
                noise_level = random.uniform(0.005, 0.03)
                audio = add_noise(audio, noise_level)
            
            # Изменяем скорость с вероятностью 50%
            if random.random() > 0.5:
                speed_factor = random.uniform(0.85, 1.15)
                audio = change_speed(audio, speed_factor)
            
            sf.write(output_file, audio, sr)
        except Exception as e:
            print(f"Ошибка при обработке файла {output_file}: {e}")
    
    print(f"Датасет успешно сгенерирован в {output_dir}")
    print(f"Создано {num_positive} позитивных и {num_negative} негативных примеров")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация датасета для обучения модели OpenWakeWord")
    parser.add_argument("--output", default="data/wake_word_dataset", help="Директория для сохранения датасета")
    parser.add_argument("--positive", type=int, default=10000, help="Количество позитивных примеров")
    parser.add_argument("--negative", type=int, default=3000, help="Количество негативных примеров")
    
    args = parser.parse_args()
    
    generate_dataset(args.output, args.positive, args.negative)