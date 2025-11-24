"""
Test des prédictions FPL
"""

import sys
import os

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predictor import quick_predict


def main():
    """Test des prédictions."""
    
    print("=" * 60)
    print("TEST DU MODULE DE PRÉDICTION")
    print("=" * 60)
    print("\nObjectifs :")
    print("  1. Charger le modèle XGBoost entraîné")
    print("  2. Charger les données des joueurs")
    print("  3. Prédire les points pour la prochaine gameweek")
    print("  4. Afficher le top 20 des joueurs recommandés")
    print("  5. Recommandations par position")
    print("  6. Sauvegarder en CSV")
    print("=" * 60)
    print()
    
    # Lancer les prédictions avec XGBoost
    df_pred = quick_predict(model_name='XGBoost', top_n=20)
    
    if df_pred is not None:
        print("\n" + "=" * 60)
        print("TEST RÉUSSI")
        print("=" * 60)
        print(f"\n{len(df_pred)} joueurs analysés")
        print("Prédictions sauvegardées dans data/predictions/latest_predictions.csv")
    else:
        print("\nErreur lors des prédictions")


if __name__ == "__main__":
    main()