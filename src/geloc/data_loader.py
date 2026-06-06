"""Загрузка и обработка данных для GeLocML"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Загрузчик данных для GeLocML.
    Поддерживает загрузку данных из текстовых файлов формата:
    имя_белка prob1 prob2 prob3 prob4 prob5 prob6 prob7 локализация
    """
    
    FEATURE_COLUMNS = ['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7']
    TARGET_COLUMN = 'localization'
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Инициализация загрузчика данных.
        
        Args:
            data_dir: Путь к директории с данными
        """
        self.data_dir = data_dir
        self.raw_data_file = os.path.join(data_dir, 'ecoli_data.txt')
    
    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, Any]]:
        """
        Парсинг одной строки файла с данными.
        
        Args:
            line: Строка с данными белка
            
        Returns:
            Dict или None если строка некорректна
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        
        parts = line.split()
        if len(parts) < 9:
            return None
        
        try:
            protein_name = parts[0]
            features = [float(x) for x in parts[1:8]]  # 7 числовых признаков
            localization = parts[8]
            
            return {
                'protein_name': protein_name,
                'prob1': features[0],
                'prob2': features[1],
                'prob3': features[2],
                'prob4': features[3],
                'prob5': features[4],
                'prob6': features[5],
                'prob7': features[6],
                'localization': localization
            }
        except (ValueError, IndexError):
            return None
    
    @staticmethod
    def read_data_file(file_path: str) -> pd.DataFrame:
        """
        Чтение файла с данными в формате ecoli.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            DataFrame с данными
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл данных не найден: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parsed = DataLoader.parse_line(line)
                if parsed is not None:
                    data.append(parsed)
                elif line.strip() and not line.startswith('#'):
                    logger.warning(f"Строка {line_num} пропущена (некорректный формат): {line.rstrip()[:50]}")
        
        if not data:
            raise ValueError(f"Не удалось загрузить данные из {file_path} - файл пуст или имеет некорректный формат")
        
        df = pd.DataFrame(data)
        logger.info(f"Загружено {len(df)} строк из {file_path}")
        logger.info(f"Классы: {sorted(df['localization'].unique())}")
        
        return df
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Загрузка обучающих данных.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Признаки (X) и целевая переменная (y)
        """
        if not os.path.exists(self.raw_data_file):
            raise FileNotFoundError(
                f"Файл с данными не найден: {self.raw_data_file}\n"
                f"Поместите файл ecoli_data.txt в директорию {self.data_dir} "
                f"или используйте скрипт генерации данных."
            )
        
        df = self.read_data_file(self.raw_data_file)
        X = df[self.FEATURE_COLUMNS]
        y = df[self.TARGET_COLUMN]
        
        logger.info(f"Обучающие данные: {len(X)} образцов, {len(X.columns)} признаков")
        logger.info(f"Распределение классов:\n{df[self.TARGET_COLUMN].value_counts()}")
        
        return X, y
    
    def load_test_data(self, test_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Загрузка тестовых данных.
        
        Args:
            test_file: Путь к тестовому файлу (если None, используется ecoli_data.txt)
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Признаки (X) и целевая переменная (y)
        """
        file_path = test_file or self.raw_data_file
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл с тестовыми данными не найден: {file_path}")
        
        df = self.read_data_file(file_path)
        X = df[self.FEATURE_COLUMNS]
        y = df[self.TARGET_COLUMN]
        
        logger.info(f"Тестовые данные: {len(X)} образцов")
        
        return X, y
    
    def load_prediction_data(self, input_path: str) -> pd.DataFrame:
        """
        Загрузка данных для предсказания.
        
        Args:
            input_path: Путь к файлу с данными для предсказания
            
        Returns:
            pd.DataFrame: Признаки для предсказания
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Файл с данными для предсказания не найден: {input_path}")
        
        df = self.read_data_file(input_path)
        X = df[self.FEATURE_COLUMNS]
        
        logger.info(f"Данные для предсказания: {len(X)} образцов")
        
        return X
    
    @staticmethod
    def save_predictions(predictions: np.ndarray, output_file: str, 
                         sample_names: Optional[List[str]] = None) -> None:
        """
        Сохранение результатов предсказания.
        
        Args:
            predictions: Массив предсказанных классов
            output_file: Путь к выходному файлу
            sample_names: Имена образцов (если есть)
        """
        import csv
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'predicted_localization'])
            
            for i, pred in enumerate(predictions):
                sample_id = sample_names[i] if sample_names else f"sample_{i+1}"
                writer.writerow([sample_id, pred])
        
        logger.info(f"Предсказания сохранены в {output_file}")
    
    @staticmethod
    def save_evaluation_results(metrics: Dict[str, float], output_file: str) -> None:
        """
        Сохранение результатов оценки модели.
        
        Args:
            metrics: Словарь с метриками
            output_file: Путь к выходному файлу
        """
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Результаты оценки модели GeLocML\n")
            f.write("=" * 40 + "\n\n")
            
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{metric_name}: {value:.4f}\n")
                else:
                    f.write(f"{metric_name}: {value}\n")
        
        logger.info(f"Результаты оценки сохранены в {output_file}")


def generate_sample_data(output_path: str = 'data/raw/ecoli_data.txt', 
                         n_samples: int = 500, random_state: int = 42) -> str:
    """
    Генерация синтетического датасета для демонстрации.
    
    Формат данных соответствует задаче прогнозирования локализации белков E. coli.
    Признаки: 7 вероятностных характеристик (prob1-prob7).
    Классы локализации: cp (цитоплазма), im (внутренняя мембрана), 
    imS (сигнальный пептид), imL (липопротеин), imU (неизвестная),
    om (внешняя мембрана), omL (липопротеин внешней мембраны), pp (периплазма).
    
    Args:
        output_path: Путь для сохранения файла
        n_samples: Количество образцов
        random_state: Seed для воспроизводимости
        
    Returns:
        str: Путь к созданному файлу
    """
    rng = np.random.RandomState(random_state)
    
    # Параметры распределений для каждого класса
    # Каждый класс имеет характерный профиль признаков
    class_profiles = {
        'cp':  {'mean': [0.8, 0.2, 0.3, 0.4, 0.2, 0.1, 0.3], 'std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        'im':  {'mean': [0.2, 0.7, 0.3, 0.2, 0.6, 0.3, 0.2], 'std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        'imS': {'mean': [0.3, 0.3, 0.8, 0.2, 0.3, 0.2, 0.5], 'std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        'imL': {'mean': [0.2, 0.3, 0.3, 0.7, 0.3, 0.2, 0.3], 'std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        'imU': {'mean': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 'std': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},
        'om':  {'mean': [0.2, 0.2, 0.2, 0.3, 0.2, 0.8, 0.2], 'std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        'omL': {'mean': [0.2, 0.3, 0.2, 0.3, 0.3, 0.3, 0.7], 'std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        'pp':  {'mean': [0.3, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2], 'std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
    }
    
    classes = list(class_profiles.keys())
    # Неравномерное распределение, как в реальных данных
    class_weights = [0.3, 0.2, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1]  # cp чаще всего
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# GeLocML - Синтетический датасет E. coli\n")
        f.write(f"# {n_samples} образцов, 7 признаков, {len(classes)} классов локализации\n")
        f.write("# protein_name prob1 prob2 prob3 prob4 prob5 prob6 prob7 localization\n\n")
        
        for i in range(n_samples):
            # Выбор класса
            cls = rng.choice(classes, p=class_weights)
            profile = class_profiles[cls]
            
            # Генерация признаков
            features = rng.normal(profile['mean'], profile['std'])
            features = np.clip(features, 0, 1)  # Вероятности в [0, 1]
            
            # Имя белка
            protein_name = f"ECO{cls}_{i+1:04d}"
            
            # Запись строки
            line = f"{protein_name} " + " ".join(f"{v:.6f}" for v in features) + f" {cls}\n"
            f.write(line)
    
    logger.info(f"Сгенерировано {n_samples} образцов в {output_path}")
    logger.info(f"Классы: {classes}")
    
    return output_path


if __name__ == '__main__':
    # Тест генерации данных
    logging.basicConfig(level=logging.INFO)
    generate_sample_data()