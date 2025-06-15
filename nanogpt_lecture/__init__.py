"""
nanogpt-lecture: Code from Neural Networks: Zero To Hero nanoGPT lecture

This package contains implementations of GPT models as demonstrated in
Andrej Karpathy's Neural Networks: Zero To Hero lecture series.
"""

from .gpt import GPTLanguageModel, GPTParams, get_batch, estimate_loss, process_input_file

__version__ = "0.1.0"
__author__ = "Andrej Karpathy"

__all__ = [
    "GPTLanguageModel", 
    "GPTParams", 
    "get_batch",
    "estimate_loss", 
    "process_input_file"
]