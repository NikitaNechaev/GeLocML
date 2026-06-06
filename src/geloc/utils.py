"""Вспомогательные функции для GeLocML"""

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
    print("  python mail.py --train --data data/raw/train.csv")
    print("  python mail.py --predict --model model.pkl --input test.fasta")
    print("  python mail.py --evaluate --model model.pkl --data data/test/")
    print()
    print("Используйте 'python mail.py --help' для просмотра всех опций")


def setup_directories():
    """Создание необходимых директорий"""
    import os
    
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
    import os
    from pathlib import Path
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Директория с данными не найдена: {data_path}")
    
    if not os.path.isdir(data_path):
        raise NotADirectoryError(f"Указанный путь не является директорией: {data_path}")


def validate_model_path(model_path):
    """Валидация пути к модели"""
    import os
    from pathlib import Path
    
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Создана директория для модели: {model_dir}")


def log_message(message, level='INFO'):
    """Логирование сообщений"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}"
    
    print(log_entry)
    
    # Сохранение в файл логов
    log_file = 'logs/geloc.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')


def load_config(config_path=None):
    """Загрузка конфигурации"""
    import json
    import os
    
    default_config = {
        'model': {
            'type': 'random_forest',
            'parameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'data': {
            'test_size': 0.2,
            'random_state': 42,
            'feature_columns': ['sequence_length', 'gc_content', 'hydrophobicity']
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Объединение с конфигурацией по умолчанию
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
            return config
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return default_config
    else:
        return default_config