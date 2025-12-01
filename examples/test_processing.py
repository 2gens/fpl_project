"""
Test du préprocessing FPL 
"""

import sys
import os

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import quick_preprocess


def main():
    
    print("Test du preprocessing des données FPL")
    
     # Lancer le preprocessing complet
    df = quick_preprocess(min_minutes=60, save=True)
    
    if df is not None:
        print("Préprocessing réussi !")
        
        # Statistiques finales
        print(f"\nStatistiques finales :")
        print(f"   • Joueurs actifs : {len(df)}")
        print(f"   • Variables : {len(df.columns)}")
        
        # Variables créées
        calculated_vars = [c for c in df.columns if '_per_' in c or '_difficulty' in c]
        print(f"   • Variables calculées : {len(calculated_vars)}")
        print(f"     {', '.join(calculated_vars[:5])}...")
        
        # Top 3 par value
        print(f"\nTOP 3 'VALUE' (points per million) :")
        top_value = df.nlargest(3, 'points_per_million')[
            ['web_name', 'team_name', 'position', 'price', 'total_points', 'points_per_million']
        ]
        for idx, row in top_value.iterrows():
            print(f"   {row['web_name']} ({row['team_name']}) - £{row['price']}M")
            print(f"      → {row['total_points']} pts = {row['points_per_million']:.2f} pts/£M")
        
        print(f"\nFichier sauvegardé : data/processed/players_cleaned.csv")
        
    else:
        print("\nÉchec du preprocessing")


if __name__ == "__main__":
    main()