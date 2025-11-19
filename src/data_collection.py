"""
Ce module gère la récupération des données depuis l'API officielle de Fantasy Premier League.
"""

import requests
import json
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List, Optional
import time


class FPLDataCollector:
    """
    Classe pour collecter les données de l'API FPL.
    """
    
    def __init__(self):
        """Initialise le collecteur de données FPL."""
        # URL de base de l'API officielle FPL
        self.base_url = "https://fantasy.premierleague.com/api"
        
        # Créer une session pour réutiliser la connexion (plus efficace)
        self.session = requests.Session()
        
        # Headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_bootstrap_static(self) -> Optional[Dict]:
        """
        Récupère les données principales de l'API FPL (bootstrap-static).
        """
        endpoint = f"{self.base_url}/bootstrap-static/"
        
        print("Connexion à l'API FPL...")
        print(f"URL: {endpoint}")
        
        try:
            # Requête GET
            response = self.session.get(endpoint, timeout=10)
            
            # Vérifier si la requête a réussi (status code 200 = OK)
            response.raise_for_status()
            
            print("Données récupérées avec succès!")
        
            data = response.json()
            
            print(f"Nombre de joueurs: {len(data['elements'])}")
            print(f"Nombre d'équipes: {len(data['teams'])}")
            print(f"Nombre de gameweeks: {len(data['events'])}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            # Si erreur
            print(f"Erreur lors de la récupération des données: {e}")
            return None
    
    def fetch_fixtures(self) -> Optional[List[Dict]]:
        """
        Récupère le calendrier des matchs (fixtures).
        """
        endpoint = f"{self.base_url}/fixtures/"
        
        print("Récupération des fixtures...")
        
        try:
            response = self.session.get(endpoint, timeout=10)
            response.raise_for_status()
            
            fixtures = response.json()
            print(f"{len(fixtures)} fixtures récupérés")
            
            return fixtures
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des fixtures: {e}")
            return None
    
    def get_player_history(self, player_id: int) -> Optional[Dict]:
        """
        Récupère l'historique détaillé d'un joueur spécifique.
        """
        endpoint = f"{self.base_url}/element-summary/{player_id}/"
        
        try:
            response = self.session.get(endpoint, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur pour le joueur {player_id}: {e}")
            return None
    
    def save_raw_data(self, data: Dict, filename: str = None) -> str:
        """
        Sauvegarde les données brutes en JSON dans data/raw/.
        """
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) 

        # Créer le dossier data/raw 
        data_dir = os.path.join(project_root, "data", "raw")
        os.makedirs(data_dir, exist_ok=True)

         # Générer un nom de fichier avec la date 
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"fpl_data_{timestamp}.json"
        
        filepath = os.path.join("data", "raw", filename)
        
        # Sauvegarder en JSON 
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Données sauvegardées: {filepath}")
        return filepath
    
    def load_players_dataframe(self, data: Dict) -> pd.DataFrame:
        """
        Convertit les données des joueurs en DataFrame pandas.
        """
        print("Conversion des données en DataFrame...")
        
        # Extraire la liste des joueurs
        players = data['elements']
        
        # Convertir en DataFrame
        df = pd.DataFrame(players)
        
        print(f"DataFrame créé: {len(df)} joueurs, {len(df.columns)} colonnes")
        
        return df
    
    def get_teams_dict(self, data: Dict) -> Dict[int, str]:
        """
        Crée un dictionnaire {team_id: team_name}.
        """
        teams = data['teams']
        return {team['id']: team['name'] for team in teams}
    
    def get_positions_dict(self, data: Dict) -> Dict[int, str]:
        """
        Crée un dictionnaire {position_id: position_name}.
         Les positions sont :
        1 = Goalkeeper (GK)
        2 = Defender (DEF)
        3 = Midfielder (MID)
        4 = Forward (FWD)
        """
        element_types = data['element_types']
        return {et['id']: et['singular_name_short'] for et in element_types}
    
    def get_next_gameweek(self, data: Dict) -> Optional[Dict]:
        """
        DÉTECTE AUTOMATIQUEMENT LA PROCHAINE GAMEWEEK À PRÉDIRE.
        """
        events = data['events']
        
        # Trouver la gameweek actuelle
        current_events = [e for e in events if e['is_current']]
        
        if not current_events:
            # Si aucune GW n'est marquée comme actuelle, prendre la première non terminée
            current_events = [e for e in events if not e['finished']]
        
        if not current_events:
            print("Attention: Toutes les gameweeks sont terminées (fin de saison?)")
            return None
        
        current = current_events[0]
        
        # Trouver la PROCHAINE gameweek (celle qu'on va prédire).
        if current['finished']:
            # La GW actuelle est finie, chercher la prochaine non terminée
            next_events = [e for e in events if e['id'] > current['id'] and not e['finished']]
        else:
            # La GW actuelle est en cours, prédire pour la suivante
            next_events = [e for e in events if e['is_next']]
        
        if not next_events:
            print("Attention: Pas de prochaine gameweek trouvée")
            return None
        
        next_gw = next_events[0]
        
        # Créer le message de statut
        if current['finished']:
            status = f"GW{current['id']} terminée → Prédiction pour GW{next_gw['id']}"
        else:
            status = f"GW{current['id']} en cours → Prédiction pour GW{next_gw['id']}"
        
        gameweek_info = {
            'current_gw': current['id'],
            'current_name': current['name'],
            'current_finished': current['finished'],
            'next_gw': next_gw['id'],
            'next_name': next_gw['name'],
            'next_deadline': next_gw['deadline_time'],
            'status': status
        }
        
        return gameweek_info
    
    def collect_all_data(self, save_raw: bool = True) -> Optional[Dict]:
        """
        Fonction principale : collecte toutes les données nécessaires.
        """
        print("=" * 60)
        print("DÉBUT DE LA COLLECTE DES DONNÉES FPL")
        print("=" * 60)
        
        # 1. Récupérer les données principales
        bootstrap_data = self.fetch_bootstrap_static()
        if bootstrap_data is None:
            print("Échec de la collecte des données principales")
            return None
        
        # 2. DÉTECTION AUTOMATIQUE DE LA GAMEWEEK
        print("\n" + "=" * 60)
        print("DÉTECTION DE LA GAMEWEEK À PRÉDIRE")
        print("=" * 60)
        gameweek_info = self.get_next_gameweek(bootstrap_data)
        if gameweek_info:
            print(f"\n{gameweek_info['status']}")
            print(f"Deadline: {gameweek_info['next_deadline']}")
        
        # 3. Récupérer les fixtures
        print("\n" + "=" * 60)
        fixtures_data = self.fetch_fixtures()
        
        # 4. Compiler toutes les données
        all_data = {
            'bootstrap': bootstrap_data,
            'fixtures': fixtures_data,
            'gameweek_info': gameweek_info,  
            'collected_at': datetime.now().isoformat()
        }
        
        # 5. Sauvegarder 
        if save_raw:
            self.save_raw_data(all_data, filename="fpl_latest.json")
        
        print("\n" + "=" * 60)
        print("COLLECTE TERMINÉE AVEC SUCCÈS!")
        print("=" * 60)
        
        return all_data



# FONCTIONS UTILITAIRES (pour faciliter l'utilisation)

def quick_collect() -> Optional[Dict]:
    """
    Fonction rapide pour collecter les données FPL.
  
    """
    collector = FPLDataCollector()
    return collector.collect_all_data(save_raw=True)


def load_latest_data() -> Optional[Dict]:
    """
    Charge les dernières données sauvegardées.
    """
    filepath = os.path.join("data", "raw", "fpl_latest.json")
    
    if not os.path.exists(filepath):
        print(f"Fichier non trouvé: {filepath}")
        print("Utilisez quick_collect() pour récupérer les données d'abord.")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Données chargées depuis {filepath}")
    
    # Afficher les infos de gameweek si disponibles
    if 'gameweek_info' in data and data['gameweek_info']:
        gw_info = data['gameweek_info']
        print(f"{gw_info['status']}")
    
    return data
