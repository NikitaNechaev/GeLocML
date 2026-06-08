"""Вспомогательные функции для GeLocML"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

__version__ = "1.0.0"
__author__ = "GeLocML Team"
__description__ = "Прогнозирование локализации белков E. coli с применением методов машинного обучения"


def print_version():
    """Печать версии программы"""
    print(f"GeLocML version {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")


def print_help():
    """Печать расширенной помощи"""
    print("GeLocML - Прогнозирование локализации белков E. coli")
    print("=" * 50)
    print()
    print("Основные режимы работы:")
    print("  --train      Обучение модели")
    print("  --predict    Предсказание локализации белков")
    print("  --evaluate   Оценка качества модели")
    print("  --version    Показать версию программы")
    print()
    print("Примеры использования:")
    print("  geloc --train --data data/raw/train.csv")
    print("  geloc --predict --model model.pkl --input test.fasta")
    print("  geloc --evaluate --model model.pkl --data data/test/")
    print()
    print("Используйте 'geloc --help' для просмотра всех опций")


def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'results',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Директория создана: {directory}")


def validate_data_path(data_path):
    """Валидация пути к данным"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Директория с данными не найдена: {data_path}")
    
    if not os.path.isdir(data_path):
        raise NotADirectoryError(f"Указанный путь не является директорией: {data_path}")


def validate_model_path(model_path):
    """Валидация пути к модели"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    
    if not os.path.isfile(model_path):
        raise ValueError(f"Указанный путь не является файлом: {model_path}")


def validate_input_path(input_path):
    """Валидация пути к входному файлу"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")
    
    if not os.path.isfile(input_path):
        raise ValueError(f"Указанный путь не является файлом: {input_path}")


def format_number(num, precision=4):
    """Форматирование числа с заданной точностью"""
    if isinstance(num, float):
        return f"{num:.{precision}f}"
    return str(num)


def safe_divide(numerator, denominator, default=0.0):
    """Безопасное деление с обработкой деления на ноль"""
    if denominator == 0:
        return default
    return numerator / denominator


def log_message(message: str, level: str = 'INFO'):
    """Логирование сообщения"""
    logger = logging.getLogger('GeLocML')
    if not logger.handlers:
        # Настройка логгера, если он еще не настроен
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    if level == 'INFO':
        logger.info(message)
    elif level == 'WARNING':
        logger.warning(message)
    elif level == 'ERROR':
        logger.error(message)
    elif level == 'DEBUG':
        logger.debug(message)
    else:
        logger.info(message)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Загрузка конфигурации из файла"""
    default_config = {
        'model': {
            'input_dim': 100,
            'hidden_dims': [128, 64, 32],
            'output_dim': 8,  # 8 классов локализации для E. coli
            'dropout_rate': 0.2
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'validation_split': 0.2
        },
        'data': {
            'sequence_length': 1000,
            'features': ['aac', 'dpc', 'entropy']
        }
    }
    
    if config_path is None:
        return default_config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError("Неподдерживаемый формат конфигурационного файла. Используйте YAML или JSON.")
        
        # Объединяем с конфигурацией по умолчанию
        def merge_dicts(default, override):
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_dicts(default_config, config)
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке конфигурации: {e}")