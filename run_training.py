#!/usr/bin/env python3
"""
GeLocML - Автоматический запуск обучения модели

Этот скрипт запускается напрямую из VSCode и сразу начинает обучение модели
на данных из папки data/. Если данных нет - генерирует синтетический датасет.
"""

import os
import sys
from pathlib import Path

# Добавляем src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from geloc.data_loader import DataLoader, generate_sample_data
from geloc.model import ProteinLocalizationModel
from geloc.utils import setup_directories, log_message


def main():
    """Основная функция - автоматический запуск обучения"""
    print("=" * 60)
    print("GeLocML - Автоматическое обучение модели локализации белков")
    print("=" * 60)
    
    # 1. Настройка директорий
    print("\n[1/5] Настройка директорий...")
    setup_directories()
    
    # 2. Проверка/генерация данных
    print("\n[2/5] Проверка данных...")
    data_dir = 'data/raw'
    data_file = os.path.join(data_dir, 'ecoli_data.txt')
    
    if not os.path.exists(data_file):
        print(f"    Файл данных не найден: {data_file}")
        print("    Генерация синтетического датасета...")
        generate_sample_data(output_path=data_file, n_samples=1000)
        print(f"    ✓ Данные сгенерированы: {data_file}")
    else:
        print(f"    ✓ Данные найдены: {data_file}")
    
    # 3. Загрузка данных
    print("\n[3/5] Загрузка данных...")
    data_loader = DataLoader(data_dir)
    X, y = data_loader.load_training_data()
    print(f"    ✓ Загружено {len(X)} образцов с {len(X.columns)} признаками")
    print(f"    ✓ Классы: {sorted(y.unique())}")
    
    # 4. Обучение модели
    print("\n[4/5] Обучение модели...")
    model = ProteinLocalizationModel()
    training_results = model.train(X, y, epochs=100, batch_size=32, learning_rate=0.001)
    
    metrics = training_results['metrics']
    print(f"    ✓ Обучение завершено!")
    print(f"    ✓ Точность на тесте: {metrics['accuracy']:.4f}")
    print(f"    ✓ CV точность: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
    
    # 5. Сохранение модели
    print("\n[5/5] Сохранение модели...")
    model_path = 'models/geloc_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"    ✓ Модель сохранена: {model_path}")
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("=" * 60)
    print(f"Модель: {model_path}")
    print(f"Обучающая выборка: {training_results['train_size']} образцов")
    print(f"Тестовая выборка: {training_results['test_size']} образцов")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"CV Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
    print("=" * 60)
    
    # Важность признаков
    feature_importance = model.get_feature_importance()
    if not feature_importance.empty:
        print("\nВажность признаков:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
