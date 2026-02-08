#!/bin/bash
# Прописываем пути к NVIDIA
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/jarvis/venv/lib/python3.12/site-packages/nvidia/cublas/lib:~/jarvis/venv/lib/python3.12/site-packages/nvidia/cudnn/lib

# Запускаем нашего Джарвиса
python3 main.py
