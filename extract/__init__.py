"""
This module contains the configuration classes for AutoGPT.
"""
from extract.extract import CharExtraction
from extract.merge import SaveEmbedding
from extract.evaluate import Evaluation
__all__ = [
    "CharExtraction",
    "SaveEmbedding",
    "Evaluation",
]