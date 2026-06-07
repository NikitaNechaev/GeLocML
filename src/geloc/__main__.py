#!/usr/bin/env python3
"""
GeLocML - Точка входа для запуска как модуля: python -m src.geloc

Автоматически запускает обучение модели на данных из папки data/
"""

import sys
import os

# Добавляем корневую директорию проекта в path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Запускаем main.py
from main import main

if __name__ == '__main__':
    main()