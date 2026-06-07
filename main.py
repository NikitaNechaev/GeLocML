#!/usr/bin/env python3
"""
GeLocML - Прогнозирование локализации белков E. coli

Основной консольный интерфейс для запуска программы.
"""

import argparse
import sys
import os
from pathlib import Path

# Добавление директории src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from geloc.core import run_geloc
from geloc.utils import print_version, print_help


def create_parser():
    """Создание парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        prog='geloc',
        description='GeLocML - Прогнозирование локализации белков E. coli с применением методов машинного обучения',
        epilog='Примеры использования:\n'
               '  python mail.py --train --data data/raw/train.csv\n'
               '  python mail.py --predict --model model.pkl --input test.fasta\n'
               '  python mail.py --version\n'
               '  python mail.py --help'
    )
    
    # Группы аргументов
    group_main = parser.add_mutually_exclusive_group()
    
    # Основные режимы работы
    group_main.add_argument('--train', action='store_true',
                          help='Запуск обучения модели')
    group_main.add_argument('--predict', action='store_true',
                          help='Запуск предсказания локализации')
    group_main.add_argument('--evaluate', action='store_true',
                          help='Оценка обученной модели')
    group_main.add_argument('--version', action='store_true',
                          help='Показать версию программы')
    
    # Общие аргументы
    parser.add_argument('--data', type=str, default='data/raw',
                      help='Путь к директории с данными (default: data/raw)')
    parser.add_argument('--model', type=str, default='models/geloc_model.pkl',
                      help='Путь к файлу модели (default: models/geloc_model.pkl)')
    parser.add_argument('--output', type=str, default='results/',
                      help='Путь к выходной директории (default: results/)')
    parser.add_argument('--config', type=str,
                      help='Путь к конфигурационному файлу')
    
    # Аргументы для обучения
    parser.add_argument('--epochs', type=int, default=100,
                      help='Количество эпох обучения (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Размер батча (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Скорость обучения (default: 0.001)')
    
    # Аргументы для предсказания
    parser.add_argument('--input', type=str,
                      help='Входной файл для предсказания')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Порог для классификации (default: 0.5)')
    
    # Уровень вывода
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Включить подробный вывод')
    parser.add_argument('-vv', '--very-verbose', action='store_true',
                      help='Включить очень подробный вывод')
    
    return parser


def handle_version(args):
    """Обработка флага --version"""
    print_version()


def handle_help(args):
    """Обработка флага --help (встроенная функция argparse)"""
    print_help()


def handle_train(args):
    """Обработка режима обучения"""
    if args.verbose:
        print("Запуск режима обучения...")
    
    try:
        from geloc.model import ProteinLocalizationModel
        
        # Путь к файлу с данными
        data_file = os.path.join(args.data, 'ecoli_data.txt')
        
        if not os.path.exists(data_file):
            print(f"Ошибка: Файл данных не найден: {data_file}")
            print("Убедитесь, что файл ecoli_data.txt находится в директории с данными")
            sys.exit(1)
        
        # Создание модели
        model = ProteinLocalizationModel()
        
        # Загрузка и обучение модели
        X, y = model.load_data(data_file)
        training_results = model.train(X, y,
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    learning_rate=args.learning_rate)
        
        # Сохранение модели
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        model.save(args.model)
        
        # Вывод результатов
        if args.verbose:
            print(f"Модель сохранена в {args.model}")
            print(f"Обучено на {training_results['train_size']} образцах")
            print(f"Точность на тестовой выборке: {training_results['metrics']['accuracy']:.4f}")
            print(f"Точность при кросс-валидации: {training_results['metrics']['cv_accuracy_mean']:.4f} (+/- {training_results['metrics']['cv_accuracy_std']:.4f})")
            
            # Важность признаков
            feature_importance = model.get_feature_importance()
            if not feature_importance.empty:
                print("\nВажность признаков:")
                for idx, row in feature_importance.iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Сохранение отчета об обучении
        report_file = os.path.join(os.path.dirname(args.model), 'training_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Отчет об обучении модели GeLocML\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Файл данных: {data_file}\n")
            f.write(f"Количество образцов: {training_results['train_size'] + training_results['test_size']}\n")
            f.write(f"Размер обучающей выборки: {training_results['train_size']}\n")
            f.write(f"Размер тестовой выборки: {training_results['test_size']}\n\n")
            
            f.write("Метрики качества:\n")
            for metric, value in training_results['metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write(f"\nКросс-валидация (5-fold):\n")
            f.write(f"  Средняя точность: {training_results['metrics']['cv_accuracy_mean']:.4f}\n")
            f.write(f"  Стандартное отклонение: {training_results['metrics']['cv_accuracy_std']:.4f}\n")
            
            # Важность признаков
            feature_importance = model.get_feature_importance()
            if not feature_importance.empty:
                f.write(f"\nВажность признаков:\n")
                for idx, row in feature_importance.iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
        
        if args.verbose:
            print(f"\nОтчет об обучении сохранен в {report_file}")
            
    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_predict(args):
    """Обработка режима предсказания"""
    if args.verbose:
        print("Запуск режима предсказания...")
    
    if not args.input:
        print("Ошибка: Укажите входной файл с помощью --input")
        sys.exit(1)
    
    try:
        from geloc.data_loader import DataLoader
        from geloc.model import ProteinLocalizationModel
        
        # Загрузка модели
        model = ProteinLocalizationModel()
        model.load(args.model)
        
        # Загрузка данных для предсказания
        data_loader = DataLoader(args.data)
        X_test = data_loader.load_prediction_data(args.input)
        
        if args.verbose:
            print(f"Загружено {len(X_test)} образцов для предсказания")
        
        # Предсказание
        predictions = model.predict(X_test, threshold=args.threshold)
        
        # Сохранение результатов
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, 'predictions.csv')
        DataLoader.save_predictions(predictions, output_file)
        
        if args.verbose:
            print(f"Результаты сохранены в {output_file}")
            
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        sys.exit(1)


def handle_evaluate(args):
    """Обработка режима оценки"""
    if args.verbose:
        print("Запуск режима оценки...")
    
    try:
        from geloc.data_loader import DataLoader
        from geloc.model import ProteinLocalizationModel
        
        # Загрузка модели
        model = ProteinLocalizationModel()
        model.load(args.model)
        
        # Загрузка тестовых данных
        data_loader = DataLoader(args.data)
        X_test, y_test = data_loader.load_test_data()
        
        if args.verbose:
            print(f"Загружено {len(X_test)} тестовых образцов")
        
        # Оценка модели
        metrics = model.evaluate(X_test, y_test)
        
        # Сохранение результатов
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, 'evaluation_results.txt')
        DataLoader.save_evaluation_results(metrics, output_file)
        
        if args.verbose:
            print(f"Результаты оценки сохранены в {output_file}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-score: {metrics['f1_score']:.4f}")
            
    except Exception as e:
        print(f"Ошибка при оценке: {e}")
        sys.exit(1)


def main():
    """Основная функция"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Обработка флага --version
    if args.version:
        handle_version(args)
        return
    
    # Обработка основных режимов
    if args.train:
        handle_train(args)
    elif args.predict:
        handle_predict(args)
    elif args.evaluate:
        handle_evaluate(args)
    else:
        # Если ни один режим не указан - автоматически запускаем обучение
        print("Режим не указан. Автоматический запуск обучения модели...")
        print("=" * 60)
        # Создаем аргументы для обучения по умолчанию
        class DefaultArgs:
            data = 'data/raw'
            model = 'models/geloc_model.pkl'
            epochs = 100
            batch_size = 32
            learning_rate = 0.001
            verbose = True
        handle_train(DefaultArgs())


if __name__ == '__main__':
    main()