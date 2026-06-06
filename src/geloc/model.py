"""
Модели машинного обучения для GeLocML
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinLocalizationModel:
    """
    Модель для прогнозирования локализации белков E. coli
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация модели
        
        Args:
            config: Конфигурация модели
        """
        self.config = config or {}
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = ['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7']
        self.target_column = 'localization'
        
        # Классы локализации
        self.classes_ = ['cp', 'im', 'imS', 'imL', 'imU', 'om', 'omL', 'pp']
        
        # Инициализация модели по умолчанию
        self._init_model()
    
    def _init_model(self):
        """Инициализация базовой модели"""
        model_params = self.config.get('parameters', {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        })
        
        self.model = OneVsRestClassifier(
            RandomForestClassifier(**model_params),
            n_jobs=-1
        )
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Загрузка и обработка данных из файла ecoli_data.txt
        
        Args:
            data_path: Путь к файлу с данными
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Признаки и целевая переменная
        """
        logger.info(f"Загрузка данных из {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл данных не найден: {data_path}")
        
        # Чтение данных
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 9:  # 7 признаков + локализация + имя белка
                    protein_name = parts[0]
                    features = [float(x) for x in parts[1:8]]  # 7 числовых признаков
                    localization = parts[8]  # Целевая переменная
                    
                    data.append({
                        'protein_name': protein_name,
                        'prob1': features[0],
                        'prob2': features[1],
                        'prob3': features[2],
                        'prob4': features[3],
                        'prob5': features[4],
                        'prob6': features[5],
                        'prob7': features[6],
                        'localization': localization
                    })
        
        if not data:
            raise ValueError("Данные не загружены - файл пуст или некорректный формат")
        
        df = pd.DataFrame(data)
        
        # Проверка на наличие всех классов локализации
        available_classes = df['localization'].unique()
        logger.info(f"Доступные классы локализации: {available_classes}")
        
        # Кодирование целевой переменной
        y = self.label_encoder.fit_transform(df['localization'])
        X = df[self.feature_columns]
        
        logger.info(f"Загружено {len(X)} образцов с {len(self.feature_columns)} признаками")
        logger.info(f"Распределение классов: {np.bincount(y)}")
        
        return X, y
    
    def preprocess_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предобработка данных
        
        Args:
            X: Признаки
            
        Returns:
            np.ndarray: Обработанные признаки
        """
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def train(self, X: pd.DataFrame, y: np.ndarray, 
               epochs: int = 100, batch_size: int = 32, 
               learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Обучение модели
        
        Args:
            X: Признаки
            y: Целевая переменная
            epochs: Количество эпох (для совместимости)
            batch_size: Размер батча (для совместимости)
            learning_rate: Скорость обучения (для совместимости)
            
        Returns:
            Dict[str, Any]: Результаты обучения
        """
        logger.info("Начало обучения модели")
        
        # Предобработка данных
        X_processed = self.preprocess_data(X)
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Размер обучающей выборки: {len(X_train)}")
        logger.info(f"Размер тестовой выборки: {len(X_test)}")
        
        # Обучение модели
        self.model.fit(X_train, y_train)
        
        # Предсказание на тестовой выборке
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Расметка метрик
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X_processed, y, cv=5, scoring='accuracy')
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
        
        logger.info("Обучение завершено")
        logger.info(f"Точность на тестовой выборке: {metrics['accuracy']:.4f}")
        logger.info(f"Точность при кросс-валидации: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
        
        return {
            'metrics': metrics,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'cv_scores': cv_scores.tolist()
        }
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Предсказание локализации белков
        
        Args:
            X: Признаки
            threshold: Порог для бинарной классификации
            
        Returns:
            np.ndarray: Предсказанные классы
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Предобработка данных
        X_processed = self.scaler.transform(X)
        
        # Предсказание
        predictions = self.model.predict(X_processed)
        
        # Преобразование обратно в исходные метки
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей
        
        Args:
            X: Признаки
            
        Returns:
            np.ndarray: Вероятности для каждого класса
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Предобработка данных
        X_processed = self.scaler.transform(X)
        
        # Предсказание вероятностей
        probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Оценка качества модели
        
        Args:
            X: Признаки
            y: Целевая переменная
            
        Returns:
            Dict[str, float]: Метрики качества
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Предсказание
        y_pred = self.model.predict(X)
        
        # Расчет метрик
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
        }
        
        return metrics
    
    def save(self, model_path: str):
        """
        Сохранение модели
        
        Args:
            model_path: Путь для сохранения модели
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Создание директорий если не существуют
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Сохранение всех компонентов модели
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'classes': self.classes_,
            'config': self.config
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Модель сохранена в {model_path}")
    
    def load(self, model_path: str):
        """
        Загрузка модели
        
        Args:
            model_path: Путь к файлу с моделью
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        # Загрузка модели
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.classes_ = model_data['classes']
        self.config = model_data['config']
        
        logger.info(f"Модель загружена из {model_path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Получение важности признаков
        
        Returns:
            pd.DataFrame: Важность признаков
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Получение важности признаков из базовой модели
        base_model = self.model.estimator_
        if hasattr(base_model, 'feature_importances_'):
            importance = base_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        else:
            logger.warning("Модель не поддерживает расчет важности признаков")
            return pd.DataFrame()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Returns:
            Dict[str, Any]: Информация о модели
        """
        return {
            'model_type': type(self.model).__name__,
            'base_estimator_type': type(self.model.estimator_).__name__,
            'n_classes': len(self.classes_),
            'n_features': len(self.feature_columns),
            'classes': self.classes_,
            'feature_columns': self.feature_columns,
            'config': self.config
        }


def create_model_from_config(config_path: str) -> ProteinLocalizationModel:
    """
    Создание модели из конфигурационного файла
    
    Args:
        config_path: Путь к конфигурационному файлу
        
    Returns:
        ProteinLocalizationModel: Созданная модель
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    return ProteinLocalizationModel(model_config)