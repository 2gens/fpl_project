"""
Comparaison ML vs Formule pondérée
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predictor import FPLPredictor


def main():
    """Comparaison des deux méthodes."""
    
    print("=" * 60)
    print("COMPARAISON : ML vs FORMULE PONDÉRÉE")
    print("=" * 60)
    print()
    
    predictor = FPLPredictor()
    
    # Charger les données
    df = predictor.load_player_data()
    if df is None:
        return
    
    print("MÉTHODE 1 : MACHINE LEARNING")
    
    # Prédictions ML
    df_ml = predictor.predict_points_ml(df)
    
    print("\nTOP 10 - MACHINE LEARNING :")
    top_ml = predictor.get_top_players(df_ml, n=10)
    predictor.print_top_players(df_ml, n=10)
    
    print("\n" + "=" * 60)
    print("MÉTHODE 2 : FORMULE PONDÉRÉE")
    print("=" * 60)
    
    # Prédictions formule
    df_formula = predictor.predict_points_formula(df)
    
    print("\nTOP 10 - FORMULE PONDÉRÉE :")
    top_formula = predictor.get_top_players(df_formula, n=10)
    predictor.print_top_players(df_formula, n=10)
    
    print("ANALYSE COMPARATIVE")
    
    # Joueurs communs dans les deux top 10
    common = set(top_ml['web_name']) & set(top_formula['web_name'])
    print(f"\nJoueurs présents dans les deux TOP 10 : {len(common)}")
    for player in common:
        print(f"  - {player}")
    
    # Sauvegarder les deux
    predictor.save_predictions(df_ml, filename="predictions_ml.csv")
    predictor.save_predictions(df_formula, filename="predictions_formula.csv")
    
    print("\n Prédictions sauvegardées :")
    print("  - data/predictions/predictions_ml.csv")
    print("  - data/predictions/predictions_formula.csv")


if __name__ == "__main__":
    main()