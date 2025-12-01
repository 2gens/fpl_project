"""
Comparaison ML vs Formule pondérée
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predictor import FPLPredictor


def main():
    
    print("Comparaison : ML vs Formule pondérée")
    predictor = FPLPredictor()
    
    # Charger les données
    df = predictor.load_player_data()
    if df is None:
        return
    
    print("Méthode 1 : Machine Learning")
    
    # Prédictions ML
    df_ml = predictor.predict_points_ml(df)
    
    print("\nTOP 10 Machine Learning :")
    top_ml = predictor.get_top_players(df_ml, n=10)

    
  
    print("Méthode 2 : Formule pondérée")
  
    # Prédictions formule
    df_formula = predictor.predict_points_formula(df)
    
    print("\nTOP 10 Formule pondérée:")
    top_formula = predictor.get_top_players(df_formula, n=10)
   
    
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