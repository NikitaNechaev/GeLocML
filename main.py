#!/usr/bin/env python3
"""
GeLocML - Прогнозирование локализации белков E. coli

Основной консольный интерфейс для запуска программы.
Этот файл является оберткой, которая добавляет src в Python path
и делегирует вызов основной функции пакета.
"""

import sys
import os

# Добавление директории src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Импорт и запуск основной функции из пакета
from geloc.__main__ import main

if __name__ == '__main__':
    main()