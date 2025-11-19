"""
Test des modèles ML FPL
"""

import sys
import os

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import train_all_models


def main():
    """Test complet des modèles ML."""
    
    print("=" * 60)
    print("TEST DES MODÈLES DE MACHINE LEARNING FPL")
    print("=" * 60)
    print("\nObjectifs :")
    print("  1. Charger les données preprocessées")
    print("  2. Entraîner les 5 modèles (LR, Ridge, RF, XGB, GB)")
    print("  3. Évaluer avec MAE, RMSE, R²")
    print("  4. Comparer les performances")
    print("  5. Afficher feature importance")
    print("  6. Sauvegarder les modèles")
    print("=" * 60)
    print()
    
    # Lancer l'entraînement complet
    trainer = train_all_models(min_minutes=60, test_size=0.2)
    
    if trainer is not None:
        print("\n" + "=" * 60)
        print("TEST RÉUSSI")
        print("=" * 60)
        print("\nRésumé :")
        print(f"  - {len(trainer.models)} modèles entraînés")
        print(f"  - {len(trainer.feature_names)} features utilisées")
        print(f"  - Modèles sauvegardés dans results/models/")
        print("\nProchaine étape : Utiliser le meilleur modèle pour faire des prédictions !")
    else:
        print("\nErreur lors de l'entraînement")


if __name__ == "__main__":
    main()