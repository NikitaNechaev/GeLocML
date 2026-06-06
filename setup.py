"""
Setup script for GeLocML - Прогнозирование локализации белков E. coli
"""

from setuptools import setup, find_packages
from pathlib import Path

# Чтение README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

setup(
    name="geloc",
    version="1.0.0",
    author="GeLocML Team",
    author_email="geloc@example.com",
    description="Прогнозирование локализации белков E. coli с применением методов машинного обучения",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/GeLocML",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "biopython>=1.79",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "pre-commit>=2.15",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "geloc=geloc.main:main",
            "geloc-train=geloc.main:train_main",
            "geloc-predict=geloc.main:predict_main",
            "geloc-evaluate=geloc.main:evaluate_main",
        ],
    },
    include_package_data=True,
    package_data={
        "geloc": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords="machine-learning bioinformatics protein-localization coli deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/username/GeLocML/issues",
        "Source": "https://github.com/username/GeLocML",
        "Documentation": "https://github.com/username/GeLocML/wiki",
    },
)