#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nanogpt-lecture",
    version="0.1.0",
    author="Andrej Karpathy",
    description="Code from Neural Networks: Zero To Hero nanoGPT lecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karpathy/ng-video-lecture",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
    ],
    entry_points={
        "console_scripts": [
            "nanogpt-train=nanogpt_lecture.gpt:main",
        ],
    },
)