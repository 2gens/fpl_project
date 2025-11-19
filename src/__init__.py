"""
Package pour la prédiction des meilleurs joueurs Fantasy Premier League.
"""

__version__ = "0.1.0"
__author__ = "Evan Dejean"

# Importation des modules principaux pour faciliter l'utilisation
from .data_collection import FPLDataCollector, quick_collect, load_latest_data

__all__ = [
    'FPLDataCollector',
    'quick_collect',
    'load_latest_data',
]