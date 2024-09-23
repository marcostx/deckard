setup(
    name="deckard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer",
    ],
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