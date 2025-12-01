"""
Test de la collecte de données FPl
"""

import sys
import os

# Ajouter le dossier parent au path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection import FPLDataCollector, quick_collect


def main():
    """Test principal du module de collecte."""
    
    print("Test de la collecte de données FPL")
    
    # Collecte des données
    print("Lancement de la collecte...")
    data = quick_collect()
    
    if not data:
        print("\nÉchec du test")
        return
    
    # Afficher les résultats
    bootstrap = data['bootstrap']
    gameweek_info = data['gameweek_info']
    
 
    print("Résumé des données collectées:")
    
    print(f"Joueurs: {len(bootstrap['elements'])}")
    print(f"Équipes: {len(bootstrap['teams'])}")
    print(f"Gameweeks: {len(bootstrap['events'])}")
    print(f"Fixtures: {len(data['fixtures'])}")
    
    print("Infos de la gameweek:")
    
    print(f"Gameweek actuelle: GW{gameweek_info['current_gw']}")
    print(f"Prochaine à prédire: GW{gameweek_info['next_gw']}")
    print(f"Status: {gameweek_info['status']}")
    print(f"Deadline: {gameweek_info['next_deadline']}")
    
    # Top 5 buteurs
    collector = FPLDataCollector()
    df = collector.load_players_dataframe(bootstrap)
    
    print("TOP 5 buteurs:")

    top_scorers = df.nlargest(5, 'goals_scored')[
        ['web_name', 'team', 'goals_scored', 'assists', 'total_points']
    ]
    print(top_scorers.to_string(index=False))
    
    # Top 5 par points totaux
    print("TOP 5 par points totaux:")
    
    top_points = df.nlargest(5, 'total_points')[
        ['web_name', 'team', 'total_points', 'now_cost', 'form']
    ]
    # Convertir le prix (divisé par 10 dans l'API)
    top_points['now_cost'] = top_points['now_cost'] / 10
    print(top_points.to_string(index=False))
    
  
    print("Test de collecte réussi !")

    print(f"\nDonnées sauvegardées dans: data/raw/fpl_latest.json")


if __name__ == "__main__":
    main()