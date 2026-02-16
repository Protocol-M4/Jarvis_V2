#!/usr/bin/env python3
"""
Скрипт для тестирования ускорения речи с сохранением высоты тона.
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import librosa.effects
import argparse

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
        print(f"speed_factor = {speed_factor}, rate = {speed_factor}")
        y_stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed_factor)
        return y_stretched
    except Exception as e:
        print(f"Ошибка при изменении скорости с librosa: {e}")
        # Запасной вариант - простой ресемплинг
        indices = np.round(np.arange(0, len(audio), 1.0/speed_factor)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]

def test_speed(input_file, output_file, speed_factor=1.2):
    """
    Тестирует изменение скорости аудио.
    
    Args:
        input_file: Путь к входному аудиофайлу
        output_file: Путь для сохранения выходного аудиофайла
        speed_factor: Коэффициент изменения скорости (>1.0 - ускорение, <1.0 - замедление)
    """
    # Загружаем аудио
    audio, sr = sf.read(input_file)
    
    # Выводим информацию о входном файле
    duration_in = len(audio) / sr
    print(f"Входной файл: {input_file}")
    print(f"Длительность: {duration_in:.2f} секунд")
    print(f"Частота дискретизации: {sr} Гц")
    print(f"Размер: {len(audio)} сэмплов")
    
    # Изменяем скорость
    audio_out = change_speed(audio, speed_factor, sr)
    
    # Выводим информацию о выходном файле
    duration_out = len(audio_out) / sr
    print(f"Выходной файл: {output_file}")
    print(f"Длительность: {duration_out:.2f} секунд")
    print(f"Размер: {len(audio_out)} сэмплов")
    
    # Вычисляем изменение длительности
    duration_change = (duration_out / duration_in - 1.0) * 100
    print(f"Изменение длительности: {duration_change:.2f}%")
    
    # Если длительность уменьшилась, значит речь ускорилась
    if duration_out < duration_in:
        print("Речь УСКОРИЛАСЬ")
    else:
        print("Речь ЗАМЕДЛИЛАСЬ")
    
    # Сохраняем выходной файл
    sf.write(output_file, audio_out, sr)
    
    print(f"Файл сохранен: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Тестирование изменения скорости аудио")
    parser.add_argument("--input", default="test_jarvis.wav", help="Путь к входному аудиофайлу")
    parser.add_argument("--output", default="test_jarvis_speed.wav", help="Путь для сохранения выходного аудиофайла")
    parser.add_argument("--speed", type=float, default=1.2, help="Коэффициент изменения скорости (>1.0 - ускорение, <1.0 - замедление)")
    
    args = parser.parse_args()
    
    test_speed(args.input, args.output, args.speed)

if __name__ == "__main__":
    main()