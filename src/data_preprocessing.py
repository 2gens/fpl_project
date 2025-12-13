"""
Ce module nettoie et prépare les données FPL pour le machine learning.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class FPLDataPreprocessor:
    
    def __init__(self, min_minutes: int = 60):
        self.min_minutes = min_minutes
        print(f"Préprocesseur initialisé")
        print(f"Filtre : minimum {min_minutes} minutes jouées")
    
    def load_raw_data(self, filepath: str = None) -> Optional[Dict]:
        """
        Charge les données brutes depuis un fichier JSON.
        """
        if filepath is None:
            
            # Chemin par défaut 
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            filepath = os.path.join(project_root, "data", "raw", "fpl_latest.json")
        
        if not os.path.exists(filepath):
            print(f"Erreur : Fichier non trouvé - {filepath}")
            return None
        
        print(f"Chargement depuis : {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Données chargées avec succès")
        return data
    
    def create_base_dataframe(self, raw_data: Dict) -> pd.DataFrame:
        """
        DataFrame de base avec tous les joueurs.
        """
        # Extraire la liste des joueurs
        players = raw_data['bootstrap']['elements']
        df = pd.DataFrame(players)
        
        print(f"DataFrame créé : {len(df)} joueurs")
        print(f"Colonnes disponibles : {len(df.columns)}")
        
        return df
    
    def filter_active_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtre pour garder seulement les joueurs ACTIFS.
        """
        # Nombre de joueurs avant filtrage
        n_before = len(df)
        print(f"Joueurs avant filtrage : {n_before}")
        
        # Filtrer : garder seulement ceux avec >= min_minutes
        df_filtered = df[df['minutes'] >= self.min_minutes].copy()
        
        # Filtrer les joueurs blessés (chance_of_playing < 50%)

        if 'chance_of_playing_this_round' in df_filtered.columns:
            n_before_injury = len(df_filtered)

            df_filtered = df_filtered[
                (
                    (df_filtered['chance_of_playing_this_round'].isna()) | 
                    (df_filtered['chance_of_playing_this_round'] >= 50)
                ) & 
                (
                (df_filtered['chance_of_playing_next_round'].isna()) | 
                (df_filtered['chance_of_playing_next_round'] >= 50)
                )
            ].copy()
        
            n_injured = n_before_injury - len(df_filtered)
            print(f"Joueurs blessés éliminés : {n_injured}")

        # Nombre de joueurs après filtrage
        n_after = len(df_filtered)
        n_removed = n_before - n_after
        
        print(f"Joueurs après filtrage : {n_after}")
        print(f"Joueurs éliminés : {n_removed} ({n_removed/n_before*100:.1f}%)")
        
        # Stats par position
        print("\nRépartition par position :")
        position_counts = df_filtered['element_type'].value_counts().sort_index()
        position_names = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        for pos_id, count in position_counts.items():
            print(f"   {position_names[pos_id]} : {count} joueurs")
        
        return df_filtered
    
    def add_position_names(self, df: pd.DataFrame, raw_data: Dict) -> pd.DataFrame:
        """
        Ajoute les noms des positions en clair (GK, DEF, MID, FWD).
        """
        # Créer le mapping position_id -> nom
        element_types = raw_data['bootstrap']['element_types']
        position_map = {et['id']: et['singular_name_short'] for et in element_types}
        
        # Ajouter la colonne position
        df['position'] = df['element_type'].map(position_map)
        
        print("Noms de positions ajoutés")
        return df
    
    def add_team_names(self, df: pd.DataFrame, raw_data: Dict) -> pd.DataFrame:
        """
        Ajoute les noms des équipes en clair (Liverpool, Man City, etc.).
        """

        # Extraire les équipes 
        teams = raw_data['bootstrap']['teams']
        team_map = {team['id']: team['name'] for team in teams}
        
        # Mapper les noms d'équipes
        df['team_name'] = df['team'].map(team_map)
        
        print("Noms d'équipes ajoutés")
        return df
    
    def engineer_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature des performances des joueurs
        """
        df = df.copy()
    
        # Convertir le coût en prix réel
        df['price'] = df['now_cost'] / 10
    
        # Goals per 90 min 
        df['goals_per_90'] = np.where(
            df['minutes'] > 0,
            (df['goals_scored'] * 90) / df['minutes'],
            0
        )
        print("Goals per 90 calculé")

        # Assists per 90 min (assist = passe décisive)
        df['assists_per_90'] = np.where(
            df['minutes'] > 0,
            (df['assists'] * 90) / df['minutes'],
            0
        )
        print("Assists per 90 calculé")
        
        # Points per game
        df['appearances'] = (df['minutes'] / 90).round()
        df['points_per_game'] = np.where(
            df['appearances'] > 0,
            df['total_points'] / df['appearances'],
            0
        )
        print("Points per game calculé")
       
        # Métrique clé pour trouver les bons plans -> Value (Points per million)
        df['points_per_million'] = df['total_points'] / df['price']
        print("Points per million calculé (Value)")
        
        # Convertir les colonnes de type string en float si nécessaire
        if df['form'].dtype == 'object':
            df['form'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
        
        if df['ict_index'].dtype == 'object':
            df['ict_index'] = pd.to_numeric(df['ict_index'], errors='coerce').fillna(0)
        
        print("Conversion des types numériques effectuée")
        
        n_new_vars = len([c for c in df.columns if c.endswith('_90') or c.endswith('_million') or c.endswith('_game')])

        # Momentum score : Détecte les joueurs en forme montante ou descendante
    
        df['momentum'] = df['form'] - df['points_per_game']
    
        print("Momentum calculé (form - points_per_game)")

        # Expected points basé sur xG et xA
        if 'expected_goals' in df.columns and 'expected_assists' in df.columns:

            df['expected_goals'] = pd.to_numeric(df['expected_goals'], errors='coerce').fillna(0)
            df['expected_assists'] = pd.to_numeric(df['expected_assists'], errors='coerce').fillna(0)

            df['expected_points_per_90'] = (
                (df['expected_goals'] * 5) + 
                (df['expected_assists'] * 3)
            ) * 90 / df['minutes'].replace(0, 1)
        
            df['expected_points_per_90'] = df['expected_points_per_90'].fillna(0)
        
            print("Expected points per 90 calculé (xG + xA)")
        else:
            df['expected_points_per_90'] = 0
            print("Attention : Expected stats non disponibles")
    

        print(f"\n{n_new_vars} nouvelles variables créées")
    
    
        return df
        
    def add_recent_minutes(self, df: pd.DataFrame, raw_data: Dict) -> pd.DataFrame:
        """
        Ajoute les minutes des 5 derniers matchs pour chaque joueur.
        """
        import requests
    
        recent_minutes_list = []
    
        for idx, row in df.iterrows():
            player_id = row['id']
        
            # Récupérer l'historique du joueur depuis l'API
            url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        
            try:
                response = requests.get(url, timeout=5)
            
                if response.status_code == 200:
                    data = response.json()
                    history = data.get('history', [])
                
                    # Trier par gameweek (plus récent d'abord)
                    history_sorted = sorted(history, key=lambda x: x['round'], reverse=True)
                
                    # Prendre les 5 derniers matchs
                    last_5 = history_sorted[:5]
                
                    # Somme des minutes
                    total_minutes_last_5 = sum(match['minutes'] for match in last_5)
                
                    recent_minutes_list.append(total_minutes_last_5)
                else:
                    recent_minutes_list.append(0)
        
            except Exception as e:
                print(f"rreur pour joueur {player_id}: {e}")
                recent_minutes_list.append(0)
    
        df['recent_5_minutes'] = recent_minutes_list
        print(f"Minutes récentes ajoutées pour {len(df)} joueurs")
    
        return df
    
    def engineer_fixture_features(self, df: pd.DataFrame, raw_data: Dict) -> pd.DataFrame:
        """
        Feature des matchs : Calculer la difficulté des prochains matchs pour chaque joueur.
        """
        print("Feature des fixtures ")
    
        
        df = df.copy()

        #Extraire les fixtures
        fixtures = raw_data['fixtures']
        gameweek_info = raw_data.get('gameweek_info', {})
        
        if not gameweek_info:
            print("Attention : Pas d'info sur la gameweek, impossible de calculer les fixtures")
            return df
        
        next_gw = gameweek_info['next_gw']
        print(f"Calcul pour GW{next_gw}")
        
        # Fixtures pour la prochaine gameweek
        next_fixtures = [f for f in fixtures if f['event'] == next_gw]
        
        print(f"{len(next_fixtures)} matchs trouvés pour GW{next_gw}")
        
        # Créer un dictionnaire {team_id: (opponent_id, difficulty, is_home)}
        team_fixtures = {}
        teams = {t['id']: t['name'] for t in raw_data['bootstrap']['teams']}
        
        for fixture in next_fixtures:
            team_h = fixture['team_h']
            team_a = fixture['team_a']
            
            # Pour l'équipe à domicile
            team_fixtures[team_h] = {
                'opponent_id': team_a,
                'opponent_name': teams.get(team_a, 'Unknown'),
                'difficulty': fixture['team_h_difficulty'],
                'is_home': True
            }
            
            # Pour l'équipe à l'extérieur
            team_fixtures[team_a] = {
                'opponent_id': team_h,
                'opponent_name': teams.get(team_h, 'Unknown'),
                'difficulty': fixture['team_a_difficulty'],
                'is_home': False
            }
        
        # Ajouter les infos au DataFrame
        df['next_fixture_difficulty'] = df['team'].map(
            lambda t: team_fixtures.get(t, {}).get('difficulty', 3)
        )
        df['next_fixture_opponent'] = df['team'].map(
            lambda t: team_fixtures.get(t, {}).get('opponent_name', 'Unknown')
        )
        df['next_fixture_home'] = df['team'].map(
            lambda t: team_fixtures.get(t, {}).get('is_home', False)
        )
        
        print("Difficulté du prochain match ajoutée")
        
        # Calculer la difficulté moyenne des 5 prochains matchs
        next_5_fixtures = [f for f in fixtures if f['event'] in range(next_gw, next_gw + 5)]
        
        team_avg_difficulty = {}
        for team_id in df['team'].unique():
            team_next_fixtures = [
                f for f in next_5_fixtures 
                if f['team_h'] == team_id or f['team_a'] == team_id
            ]
            
            difficulties = []
            for f in team_next_fixtures:
                if f['team_h'] == team_id:
                    difficulties.append(f['team_h_difficulty'])
                else:
                    difficulties.append(f['team_a_difficulty'])
            
            if difficulties:
                team_avg_difficulty[team_id] = np.mean(difficulties)
            else:
                team_avg_difficulty[team_id] = 3  
        
        df['avg_fixture_difficulty_5'] = df['team'].map(team_avg_difficulty)
        
        print("Difficulté moyenne sur 5 matchs ajoutée")

        # Recent Performance Score -> Combine form récente, expected stats et fiabilité
        df['recent_performance_score'] = (
            df['form'] * 0.5 +                      
            df['expected_points_per_90'] * 0.3 +   
            df['points_per_game'] * 0.2          
        )
    
        print("Recent performance score calculé (target pour ML)")

      
        # Expected minutes (xM) - avec momentum
        
        # 1. Temps de jeu historique (total minutes jouées)
        max_possible_minutes = df['minutes'].max()
        df['historical_minutes_ratio'] = (df['minutes'] / max_possible_minutes).clip(upper=1.0)

        # 2. Temps de jeu récent (5 derniers matchs)
        df['recent_5_minutes_ratio'] = (df['recent_5_minutes'] / 450).clip(upper=1.0)  # 5 matchs × 90 min

        # 3. xM final = Moyenne pondérée (70% récent, 30% historique)
        df['xM_factor'] = (
    df['recent_5_minutes_ratio'] * 0.7 +
    df['historical_minutes_ratio'] * 0.3
).clip(lower=0.3, upper=1.0)

        print(f" xM avec momentum appliqué (données réelles !)")
        print(f" Joueurs avec xM < 0.5 (rotation risk) : {(df['xM_factor'] < 0.5).sum()}")
        print(f" Moyenne xM_factor : {df['xM_factor'].mean():.2f}")

        # Team strength (home/away)

        # Extraire les forces d'attaque et de défense de chaque équipe
        teams = raw_data['bootstrap']['teams']

        # Créer les mappings team_id → strength
        team_attack_home = {t['id']: t['strength_attack_home'] for t in teams}
        team_attack_away = {t['id']: t['strength_attack_away'] for t in teams}
        team_defence_home = {t['id']: t['strength_defence_home'] for t in teams}
        team_defence_away = {t['id']: t['strength_defence_away'] for t in teams}

        # Calculer la moyenne de la ligue
        league_avg_attack = sum(team_attack_home.values()) / len(team_attack_home)
        league_avg_defence = sum(team_defence_home.values()) / len(team_defence_home)

        print(f"   Moyenne ligue - Attaque: {league_avg_attack:.0f}, Défense: {league_avg_defence:.0f}")

        # Créer mapping nom → ID
        team_name_to_id = {t['name']: t['id'] for t in teams}
        df['opponent_team_id'] = df['next_fixture_opponent'].map(team_name_to_id)

        # Calculer l'ajustement intelligent

        def calculate_adjustment_for_row(row):
    
            opponent_id = row['opponent_team_id']
    
            # Si pas d'adversaire connu, pas d'ajustement
            if pd.isna(opponent_id):
                return 1.0
    
            opponent_id = int(opponent_id)
            is_home = row['next_fixture_home']
            position = row['position']
    
            # Récupérer les forces de l'adversaire
            if is_home:
                # Le joueur est à domicile → adversaire est away
                opponent_attack = team_attack_away.get(opponent_id, league_avg_attack)
                opponent_defence = team_defence_away.get(opponent_id, league_avg_defence)
            else:
                # Le joueur est away → adversaire est home
                opponent_attack = team_attack_home.get(opponent_id, league_avg_attack)
                opponent_defence = team_defence_home.get(opponent_id, league_avg_defence)
    
            # Calculer ajustement selon position
            if position in ['FWD', 'MID']:
                # FWD et MID : Impactés par la défense adverse
                adjustment = league_avg_defence / opponent_defence
    
            elif position == 'DEF':
                # DEF : Impactés par l'attaque adverse (clean sheet)
                adjustment = league_avg_attack / opponent_attack
    
            elif position == 'GK':
                # GK : Très impactés par l'attaque adverse
                adjustment = league_avg_attack / opponent_attack 
    
            else:
                adjustment = 1.0
    
            # Clipper pour éviter extrêmes
            return max(0.7, min(1.3, adjustment))

        # Appliquer la fonction à chaque ligne du DataFrame
        df['smart_adjustment'] = df.apply(calculate_adjustment_for_row, axis=1)

        return df
    
    def select_important_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sélectionne uniquement les colonnes importantes pour le ML.
        """

        # Colonnes à garder
        important_cols = [
            # Identité
            'id', 'web_name', 'first_name', 'second_name',
            'team', 'team_name', 'element_type', 'position',
            
            # Prix et sélection
            'price', 'now_cost', 'selected_by_percent',
            
            # Stats de base
            'total_points', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'goals_conceded', 'own_goals',
            'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards',
            'saves', 'bonus', 'bps',

            # Disponibilité 
            'chance_of_playing_this_round', 'chance_of_playing_next_round',
            
            # Forme et indices
            'form', 'points_per_game', 'ict_index',
            'influence', 'creativity', 'threat', 'momentum',
            
            # Set pieces
            'penalties_order', 'corners_and_indirect_freekicks_order',
            'direct_freekicks_order',
            
            # Expected stats 
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'expected_goals_conceded','expected_points_per_90',
            
            # Variables calculées (feature engineering)
            'goals_per_90', 'assists_per_90', 'points_per_million','recent_performance_score',
            
            # Fixtures
            'next_fixture_difficulty', 'next_fixture_opponent', 
            'next_fixture_home', 'avg_fixture_difficulty_5',

            #Features avancées pour le ML 
            'xM_factor', 'recent_5_minutes', 'recent_5_minutes_ratio', 
            'historical_minutes_ratio', 'smart_adjustment', 'opponent_team_id', 
        ]
        
        # Garder seulement les colonnes qui existent
        available_cols = [col for col in important_cols if col in df.columns]
        
        df_selected = df[available_cols].copy()
        
        print(f"Colonnes sélectionnées : {len(available_cols)} / {len(important_cols)}")
        print(f"DataFrame final : {len(df_selected)} joueurs × {len(df_selected.columns)} colonnes")
        
        return df_selected
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Sauvegarde les données nettoyées en CSV dans data/processed/.
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        data_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(data_dir, exist_ok=True)
        
        # Générer un nom de fichier
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"players_processed_{timestamp}.csv"
        
        filepath = os.path.join(data_dir, filename)
        
        # Sauvegarder en CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"\nDonnées sauvegardées : {filepath}")
        print(f"{len(df)} joueurs × {len(df.columns)} colonnes")
        
        return filepath


# Fonction utilitaire

def quick_preprocess(min_minutes: int = 60, save: bool = True) -> Optional[pd.DataFrame]:
  
    preprocessor = FPLDataPreprocessor(min_minutes=min_minutes)
    
    # Charger les données brutes
    raw_data = preprocessor.load_raw_data()
    if raw_data is None:
        return None
    
    # Créer le DataFrame de base
    df = preprocessor.create_base_dataframe(raw_data)
    
    # Filtrer les joueurs actifs
    df = preprocessor.filter_active_players(df)
    
    # Ajouter les noms (positions et équipes)
    df = preprocessor.add_position_names(df, raw_data)
    df = preprocessor.add_team_names(df, raw_data)
    
    # Feature Performances
    df = preprocessor.engineer_performance_features(df)

    # Ajouter les minutes récentes -> des 5 derniers matchs
    df = preprocessor.add_recent_minutes(df, raw_data)
    
    # Feature Fixtures
    df = preprocessor.engineer_fixture_features(df, raw_data)
    
    # Sélectionner les colonnes importantes
    df = preprocessor.select_important_columns(df)
    
    # Sauvegarder les données nettoyées
    if save:
        print("Sauvegarde des données nettoyées...")
        preprocessor.save_processed_data(df, filename="players_cleaned.csv")
    
    print("Preprocessing terminé.")
    
    return df