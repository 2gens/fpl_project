"""
Pipeline FPL Complet
"""

import sys
import os
import time
from datetime import datetime

# Ajouter src au path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_collection import quick_collect
from src.data_preprocessing import quick_preprocess
from src.models import train_all_models
from src.eda import quick_eda
from src.predictor import quick_predict


def print_header(title: str):
    
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, step_name: str):
    
    print(f"  ÉTAPE {step_num}/6 : {step_name}")
 


def main():
    
    start_time = time.time()
    
    print_header("PIPELINE FPL COMPLET")
    print(f"\nDate et heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nCe script va exécuter toutes les étapes du projet :")
    print("  1. Collecte des données")
    print("  2. Preprocessing")
    print("  3. Entraînement ML")
    print("  4. Génération graphiques EDA")
    print("  5. Prédictions ML")
    print("  6. Prédictions Formule")
    print("  7. Comparaison des méthodes")
    
    input("\nAppuyez sur ENTRÉE pour commencer...")
    
    # Etape 1 : Collecte des données
    print_step(1, "Collecte des données")
    
    try:
        data = quick_collect()
        if data is None:
            print("\nERREUR : Collecte des données échouée")
            return
        print("\nCollecte terminée avec succès")
    except Exception as e:
        print(f"\nERREUR lors de la collecte : {e}")
        return
    
    # Etape 2 : Preprocessing et Feature Engineering
    print_step(2, "Preprocessing et Feature Engineering")
    
    try:
        df = quick_preprocess(min_minutes=60, save=True)
        if df is None:
            print("\nERREUR : Preprocessing échoué")
            return
        print("\nPreprocessing terminé avec succès")
    except Exception as e:
        print(f"\nERREUR lors du preprocessing : {e}")
        return
    
    # Etape 3 : Entraînement des modèles ML
    print_step(3, "Entraînement des modèles ML")
    
    try:
        results = train_all_models()
        if results is None:
            print("\nERREUR : Entraînement échoué")
            return
        print("\nEntraînement terminé avec succès")
    except Exception as e:
        print(f"\nERREUR lors de l'entraînement : {e}")
        return
    
    # Etape 4 : Génération des graphiques EDA
    print_step(4, "Génération des graphiques EDA")
    
    try:
        eda = quick_eda()
        print("\nGraphiques générés avec succès")
    except Exception as e:
        print(f"\nERREUR lors de la génération EDA : {e}")
       
    # Etape 5 et 6 : Prédictions ML et Formule pondérée

    print_step(5, "PRÉDICTIONS")

    from src.predictor import FPLPredictor

    predictor = FPLPredictor()
    df = predictor.load_player_data()

    df_ml = predictor.predict_points_ml(df)

    if df_ml is not None:
        predictor.save_predictions(df_ml, filename="predictions_ml.csv")
        print("Prédictions ML sauvegardées")

    df_formula = predictor.predict_points_formula(df)

    if df_formula is not None:
        predictor.save_predictions(df_formula, filename="predictions_formula.csv")
        print("Prédictions Formule sauvegardées")
    
    
    # Etape 7 : Comparaison des deux méthodes
    print_step(7, "Comparaison des méthodes")
    
    if df_ml is not None and df_formula is not None:
        try:
            # Top 20 de chaque
            top_ml = df_ml.nlargest(20, 'predicted_points')['web_name'].tolist()
            top_formula = df_formula.nlargest(20, 'predicted_points')['web_name'].tolist()
            
            # Joueurs communs
            common = set(top_ml) & set(top_formula)
            
        
            print("Analyse comparative des TOP 20 :")
            
            print(f"\nJoueurs présents dans les deux TOP 30 : {len(common)}/20")
            
            coherence = (len(common) / 20) * 100
            print(f"Cohérence entre les méthodes : {coherence:.0f}%")
            
            
            print("\nJoueurs communs :")
            for player in sorted(common):
                print(f" {player}")
            
            print("\nComparaison terminée")
        except Exception as e:
            print(f"\nERREUR lors de la comparaison : {e}")
    else:
        print("\nComparaison impossible (une des méthodes a échoué)")
    
    # Fin
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print(" Pipeline FPL terminée ")
    print("=" * 70)
    
    
    print("\nFICHIERS GÉNÉRÉS :")
    print("  ├── data/raw/fpl_latest.json")
    print("  ├── data/processed/players_cleaned.csv")
    print("  ├── data/predictions/predictions_ml.csv")
    print("  ├── data/predictions/predictions_formula.csv")
    print("  ├── results/models/ (5 modèles .pkl)")
    print("  ├── results/figures/ (7 graphiques .png)")
    print("  └── results/models/results_summary.json")


if __name__ == "__main__":
    main()