# Makefile для GeLocML

.PHONY: help install install-dev test clean lint format build upload docs

# Цвета для вывода
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Переменные
VENV_NAME = venv
PYTHON = python3
PIP = pip

help: ## Показать справку по Makefile
	@echo "$(BLUE)GeLocML Makefile$(NC)"
	@echo "$(YELLOW)Доступные команды:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Установить зависимости проекта
	@echo "$(BLUE)Установка зависимостей...$(NC)"
	$(PIP) install -r requirements.txt

install-dev: ## Установить зависимости для разработки
	@echo "$(BLUE)Установка зависимостей для разработки...$(NC)"
	$(PIP) install -e .[dev]

install-all: install-dev ## Установить все зависимости (основные + разработка)
	@echo "$(BLUE)Установка всех зависимостей...$(NC)"

venv: ## Создать виртуальное окружение
	@echo "$(BLUE)Создание виртуального окружения...$(NC)"
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "$(GREEN)Виртуальное окружение создано: $(VENV_NAME)$(NC)"
	@echo "$(YELLOW)Активируйте его: source $(VENV_NAME)/bin/activate$(NC)"

test: ## Запустить тесты
	@echo "$(BLUE)Запуск тестов...$(NC)"
	pytest tests/ -v

test-coverage: ## Запустить тесты с покрытием
	@echo "$(BLUE)Запуск тестов с покрытием...$(NC)"
	pytest tests/ --cov=geloc --cov-report=html --cov-report=term

lint: ## Запустить линтинг
	@echo "$(BLUE)Запуск линтинга...$(NC)"
	flake8 src/geloc/
	@echo "$(GREEN)Линтинг завершен$(NC)"

format: ## Форматировать код
	@echo "$(BLUE)Форматирование кода...$(NC)"
	black src/geloc/ tests/
	@echo "$(GREEN)Код отформатирован$(NC)"

format-check: ## Проверить форматирование кода
	@echo "$(BLUE)Проверка форматирования кода...$(NC)"
	black --check src/geloc/ tests/
	@echo "$(GREEN)Форматирование корректно$(NC)"

build: ## Сборка проекта
	@echo "$(BLUE)Сборка проекта...$(NC)"
	python setup.py sdist bdist_wheel

upload: ## Загрузить в PyPI
	@echo "$(BLUE)Загрузка в PyPI...$(NC)"
	twine upload dist/*

clean: ## Очистить артефакты сборки
	@echo "$(BLUE)Очистка артефактов...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Очистка завершена$(NC)"

docs: ## Сгенерировать документацию
	@echo "$(BLUE)Генерация документации...$(NC)"
	cd docs && make html
	@echo "$(GREEN)Документация сгенерирована в docs/_build/html$(NC)"

run: ## Запустить GeLoc (пример)
	@echo "$(BLUE)Запуск GeLoc...$(NC)"
	python mail.py --help

run-train: ## Запустить обучение
	@echo "$(BLUE)Запуск обучения...$(NC)"
	python mail.py --train --data data/raw/

run-predict: ## Запустить предсказание
	@echo "$(BLUE)Запуск предсказания...$(NC)"
	python mail.py --predict --model models/geloc_model.pkl --input data/test.fasta

check-deps: ## Проверить зависимости
	@echo "$(BLUE)Проверка зависимостей...$(NC)"
	pip check

setup-dev: ## Настройка окружения для разработки
	@echo "$(BLUE)Настройка окружения для разработки...$(NC)"
	$(MAKE) venv
	$(MAKE) install-dev
	$(MAKE) format-check
	@echo "$(GREEN)Настройка завершена!$(NC)"
	@echo "$(YELLOW)Активируйте виртуальное окружение: source $(VENV_NAME)/bin/activate$(NC)"

# Алиасы для удобства
install-deps: install
dev-setup: setup-dev
test-all: test test-coverage

# Правило для отображения версии
version: ## Показать версию проекта
	@echo "$(BLUE)Версия GeLocML:$(NC)"
	@python -c "import sys; sys.path.insert(0, 'src'); from geloc.utils import __version__; print(__version__)"