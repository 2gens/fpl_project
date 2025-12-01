"""
Test de l'analyse EDA
"""

import sys
import os

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda import quick_eda


def main():
    """Test de l'EDA."""
    
    print("Test de l'annalyse exploratoire (EDA)")
    print("\nGraphiques à générer :")
    print("  1. Distribution des points par position")
    print("  2. Top 15 joueurs par points")
    print("  3. Prix vs Points (scatter plot)")
    print("  4. Matrice de corrélation")
    print("  5. Goals/Assists per 90 par position")
    print("  6. Top 15 meilleurs value players")
    print("  7. Distribution fixture difficulty")
 
    
    # Générer tous les graphiques
    eda = quick_eda()
    
    print("Test réussi !")
    print("\nFichiers créés :")
    print("  - points_distribution_by_position.png")
    print("  - top_15_players.png")
    print("  - price_vs_points.png")
    print("  - correlation_matrix.png")
    print("  - goals_assists_per_90.png")
    print("  - top_15_value_players.png")
    print("  - fixture_difficulty_distribution.png")


if __name__ == "__main__":
    main()