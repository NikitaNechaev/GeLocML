"""Основной модуль логики GeLocML"""

import os
import sys
from typing import Dict, Any, Optional
from .utils import log_message, setup_directories, validate_data_path, validate_model_path, load_config


def run_geloc(mode: str, **kwargs) -> Dict[str, Any]:
    """
    Основная функция для запуска GeLoc
    
    Args:
        mode: Режим работы ('train', 'predict', 'evaluate')
        **kwargs: Дополнительные параметры
        
    Returns:
        Dict[str, Any]: Результаты выполнения
    """
    try:
        # Настройка директорий
        setup_directories()
        
        # Загрузка конфигурации
        config = load_config(kwargs.get('config'))
        
        log_message(f"Запуск GeLoc в режиме: {mode}")
        
        # Валидация данных
        if 'data' in kwargs:
            validate_data_path(kwargs['data'])
        
        # Валидация пути к модели
        if 'model' in kwargs:
            validate_model_path(kwargs['model'])
        
        # Выбор режима работы
        if mode == 'train':
            return train_model(config, **kwargs)
        elif mode == 'predict':
            return predict_localization(config, **kwargs)
        elif mode == 'evaluate':
            return evaluate_model(config, **kwargs)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")
            
    except Exception as e:
        log_message(f"Ошибка при запуске GeLoc: {str(e)}", 'ERROR')
        raise


def train_model(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Обучение модели
    
    Args:
        config: Конфигурация
        **kwargs: Параметры обучения
        
    Returns:
        Dict[str, Any]: Результаты обучения
    """
    log_message("Начало обучения модели")
    
    try:
        from .data_loader import DataLoader
        from .model import ProteinLocalizationModel
        
        # Загрузка данных
        data_loader = DataLoader(kwargs.get('data', 'data/raw'))
        X_train, y_train = data_loader.load_training_data()
        
        log_message(f"Загружено {len(X_train)} образцов для обучения")
        
        # Создание и обучение модели
        model = ProteinLocalizationModel(config['model'])
        training_history = model.train(
            X_train, y_train,
            epochs=kwargs.get('epochs', config['training']['epochs']),
            batch_size=kwargs.get('batch_size', config['training']['batch_size']),
            learning_rate=kwargs.get('learning_rate', config['training']['learning_rate'])
        )
        
        # Сохранение модели
        model_path = kwargs.get('model', 'models/geloc_model.pkl')
        model.save(model_path)
        
        log_message(f"Модель сохранена в {model_path}")
        
        return {
            'success': True,
            'model_path': model_path,
            'training_history': training_history,
            'samples_count': len(X_train)
        }
        
    except Exception as e:
        log_message(f"Ошибка при обучении модели: {str(e)}", 'ERROR')
        raise


def predict_localization(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Предсказание локализации белков
    
    Args:
        config: Конфигурация
        **kwargs: Параметры предсказания
        
    Returns:
        Dict[str, Any]: Результаты предсказания
    """
    log_message("Начало предсказания локализации")
    
    if not kwargs.get('input'):
        raise ValueError("Не указан входной файл (--input)")
    
    try:
        from .data_loader import DataLoader
        from .model import ProteinLocalizationModel
        
        # Загрузка модели
        model = ProteinLocalizationModel(config['model'])
        model_path = kwargs.get('model', 'models/geloc_model.pkl')
        model.load(model_path)
        
        # Загрузка данных
        data_loader = DataLoader(kwargs.get('data', 'data/raw'))
        X_test = data_loader.load_prediction_data(kwargs['input'])
        
        log_message(f"Загружено {len(X_test)} образцов для предсказания")
        
        # Предсказание
        predictions = model.predict(X_test, threshold=kwargs.get('threshold', 0.5))
        
        # Сохранение результатов
        output_dir = kwargs.get('output', 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'predictions.csv')
        data_loader.save_predictions(predictions, output_file)
        
        log_message(f"Результаты сохранены в {output_file}")
        
        return {
            'success': True,
            'predictions_file': output_file,
            'samples_count': len(X_test),
            'predictions': predictions
        }
        
    except Exception as e:
        log_message(f"Ошибка при предсказании локализации: {str(e)}", 'ERROR')
        raise


def evaluate_model(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Оценка качества модели
    
    Args:
        config: Конфигурация
        **kwargs: Параметры оценки
        
    Returns:
        Dict[str, Any]: Результаты оценки
    """
    log_message("Начало оценки модели")
    
    try:
        from .data_loader import DataLoader
        from .model import ProteinLocalizationModel
        
        # Загрузка модели
        model = ProteinLocalizationModel(config['model'])
        model_path = kwargs.get('model', 'models/geloc_model.pkl')
        model.load(model_path)
        
        # Загрузка тестовых данных
        data_loader = DataLoader(kwargs.get('data', 'data/raw'))
        X_test, y_test = data_loader.load_test_data()
        
        log_message(f"Загружено {len(X_test)} тестовых образцов")
        
        # Оценка модели
        metrics = model.evaluate(X_test, y_test)
        
        # Сохранение результатов
        output_dir = kwargs.get('output', 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'evaluation_results.txt')
        data_loader.save_evaluation_results(metrics, output_file)
        
        log_message(f"Результаты оценки сохранены в {output_file}")
        log_message(f"Accuracy: {metrics['accuracy']:.4f}")
        log_message(f"Precision: {metrics['precision']:.4f}")
        log_message(f"Recall: {metrics['recall']:.4f}")
        log_message(f"F1-score: {metrics['f1_score']:.4f}")
        
        return {
            'success': True,
            'metrics_file': output_file,
            'metrics': metrics,
            'samples_count': len(X_test)
        }
        
    except Exception as e:
        log_message(f"Ошибка при оценке модели: {str(e)}", 'ERROR')
        raise


def get_system_info() -> Dict[str, Any]:
    """
    Получение информации о системе
    
    Returns:
        Dict[str, Any]: Информация о системе
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent
    }


def health_check() -> Dict[str, Any]:
    """
    Проверка состояния системы
    
    Returns:
        Dict[str, Any]: Результаты проверки
    """
    try:
        system_info = get_system_info()
        
        # Проверка доступной памяти
        memory_available_gb = system_info['memory_available'] / (1024**3)
        memory_ok = memory_available_gb > 1  # минимум 1 ГБ
        
        # Проверка дискового пространства
        disk_ok = system_info['disk_usage'] < 90  # меньше 90% использовано
        
        health_status = {
            'overall': memory_ok and disk_ok,
            'memory_available_gb': memory_available_gb,
            'memory_ok': memory_ok,
            'disk_usage_percent': system_info['disk_usage'],
            'disk_ok': disk_ok,
            'system_info': system_info
        }
        
        log_message(f"Проверка состояния системы: {'✓' if health_status['overall'] else '✗'}")
        
        return health_status
        
    except Exception as e:
        log_message(f"Ошибка при проверке состояния системы: {str(e)}", 'ERROR')
        return {'overall': False, 'error': str(e)}