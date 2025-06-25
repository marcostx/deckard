from setuptools import setup, find_packages

setup(
    name="deckard",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "typer>=0.9.0",
        "joblib>=1.2.0",
        "shap>=0.42.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "flask>=2.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deckard=deckard.cli:app",
        ],
    },
    author="Author",
    description="A CLI library to develop, manage, deploy and monitor machine learning models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/marcostx/deckard",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)