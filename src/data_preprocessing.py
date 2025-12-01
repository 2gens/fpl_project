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
    """
    Classe pour prétraiter les données FPL.
    """
    
    def __init__(self, min_minutes: int = 60):
        self.min_minutes = min_minutes
        print(f"Préprocesseur initialisé")
        print(f"Filtre : minimum {min_minutes} minutes jouées")
    
    def load_raw_data(self, filepath: str = None) -> Optional[Dict]:
        """
        Charge les données brutes depuis un fichier JSON.
        """
        if filepath is None:
            # Trouver le dossier racine du projet
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
        Crée le DataFrame de base avec tous les joueurs.
        """
        print("\n" + "=" * 60)
        print("CRÉATION DU DATAFRAME DE BASE")
        print("=" * 60)
        
        # Extraire les joueurs
        players = raw_data['bootstrap']['elements']
        df = pd.DataFrame(players)
        
        print(f"DataFrame créé : {len(df)} joueurs")
        print(f"Colonnes disponibles : {len(df.columns)}")
        
        return df
    
    def filter_active_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtre pour garder seulement les joueurs ACTIFS.
        """
        print("\n" + "=" * 60)
        print(f"FILTRAGE DES JOUEURS ACTIFS (>= {self.min_minutes} min)")
        print("=" * 60)
        
        # Nombre de joueurs avant filtrage
        n_before = len(df)
        print(f"Joueurs avant filtrage : {n_before}")
        
        # Filtrer : garder seulement ceux avec >= min_minutes
        df_filtered = df[df['minutes'] >= self.min_minutes].copy()
        
        # Filtrer les joueurs blessés (chance_of_playing < 25%)
        if 'chance_of_playing_this_round' in df_filtered.columns:
            n_before_injury = len(df_filtered)
            df_filtered = df_filtered[
            (df_filtered['chance_of_playing_this_round'].isna()) | 
            (df_filtered['chance_of_playing_this_round'] >= 25)].copy()
        
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
        # Créer le mapping team_id -> nom
        teams = raw_data['bootstrap']['teams']
        team_map = {team['id']: team['name'] for team in teams}
        
        # Ajouter la colonne team_name
        df['team_name'] = df['team'].map(team_map)
        
        print("Noms d'équipes ajoutés")
        return df
    
    def show_sample(self, df: pd.DataFrame, n: int = 10):
        """
        Affiche un échantillon du DataFrame pour vérification.
        """
        print("\n" + "=" * 60)
        print(f"APERÇU DES DONNÉES (Top {n} par points totaux)")
        print("=" * 60)
        
        # Colonnes importantes à afficher
        cols_to_show = [
            'web_name', 'team_name', 'position', 
            'total_points', 'minutes', 'goals_scored', 
            'assists', 'form', 'now_cost'
        ]
        
        # Sélectionner les colonnes qui existent
        available_cols = [col for col in cols_to_show if col in df.columns]
        
        # Trier par points totaux et afficher
        sample = df.nlargest(n, 'total_points')[available_cols]
        
        # Convertir le prix (divisé par 10 dans l'API)
        if 'now_cost' in sample.columns:
            sample = sample.copy()
            sample['now_cost'] = sample['now_cost'] / 10
        
        print(sample.to_string(index=False))
        print()
    
    def engineer_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FEATURE ENGINEERING - PARTIE 1 : PERFORMANCES
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING - PERFORMANCES")
        print("=" * 60)
        
        df = df.copy()
        
        # 1. Prix en £M (plus lisible que divisé par 10)
        df['price'] = df['now_cost'] / 10
        print("Prix converti en £M")
        
        # 2. Goals per 90 minutes
        # Formule : (goals * 90) / minutes_jouées
        # Si 0 minutes, mettre 0
        df['goals_per_90'] = np.where(
            df['minutes'] > 0,
            (df['goals_scored'] * 90) / df['minutes'],
            0
        )
        print("Goals per 90 calculé")
        
        # 3. Assists per 90 minutes
        df['assists_per_90'] = np.where(
            df['minutes'] > 0,
            (df['assists'] * 90) / df['minutes'],
            0
        )
        print("Assists per 90 calculé")
        
        # 4. Points per game (moyenne par match joué)
        # Un match = environ 90 minutes, donc appearances ≈ minutes / 90
        df['appearances'] = (df['minutes'] / 90).round()
        df['points_per_game'] = np.where(
            df['appearances'] > 0,
            df['total_points'] / df['appearances'],
            0
        )
        print("Points per game calculé")
        
        # 5. Points per million (VALUE)
        # Métrique clé pour trouver les bons plans
        df['points_per_million'] = df['total_points'] / df['price']
        print("Points per million calculé (VALUE)")
        
        # 6. Convertir les colonnes de type string en float si nécessaire
        # (form et ict_index sont parfois en string dans l'API)
        if df['form'].dtype == 'object':
            df['form'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
        
        if df['ict_index'].dtype == 'object':
            df['ict_index'] = pd.to_numeric(df['ict_index'], errors='coerce').fillna(0)
        
        print("Conversion des types numériques effectuée")
        
        n_new_vars = len([c for c in df.columns if c.endswith('_90') or c.endswith('_million') or c.endswith('_game')])

        # 7. MOMENTUM SCORE - Détecte les joueurs en forme montante -> ii form > points_per_game → le joueur est en forme montante 
    
        df['momentum'] = df['form'] - df['points_per_game']
    
        print("Momentum calculé (form - points_per_game)")

        # 8. Expected points basé sur xG et xA
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
        
    
    def engineer_fixture_features(self, df: pd.DataFrame, raw_data: Dict) -> pd.DataFrame:
        """
        FEATURE ENGINEERING - PARTIE 2 : FIXTURES (DIFFICULTÉ)      
        Calcule la difficulté des prochains matchs pour chaque joueur.
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING - FIXTURES & DIFFICULTÉ")
        print("=" * 60)
        
        df = df.copy()
        fixtures = raw_data['fixtures']
        gameweek_info = raw_data.get('gameweek_info', {})
        
        if not gameweek_info:
            print("Attention : Pas d'info sur la gameweek, impossible de calculer les fixtures")
            return df
        
        next_gw = gameweek_info['next_gw']
        print(f"Calcul pour GW{next_gw}")
        
        # Filtrer les fixtures pour la prochaine gameweek
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
                team_avg_difficulty[team_id] = 3  # Moyenne par défaut
        
        df['avg_fixture_difficulty_5'] = df['team'].map(team_avg_difficulty)
        
        print("Difficulté moyenne sur 5 matchs ajoutée")

         # Recent Performance Score -> Combine form récente, expected stats et fiabilité
        df['recent_performance_score'] = (
            df['form'] * 0.5 +                      # 50% forme récente 
            df['expected_points_per_90'] * 0.3 +    # 30% expected stats (xG/xA)
            df['points_per_game'] * 0.2             # 20% fiabilité saison
        )
    
        # Ajuster selon la difficulté du prochain match
        fixture_adjustment = df['next_fixture_difficulty'].apply(
            lambda x: 1.2 if x <= 2 else 1.0 if x == 3 else 0.85
        )
    
        df['recent_performance_score'] = df['recent_performance_score'] * fixture_adjustment
    
        print("Recent performance score calculé (target pour ML)")
        
        return df
    
    def select_important_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sélectionne uniquement les colonnes importantes pour le ML.
        """
        print("\n" + "=" * 60)
        print("SÉLECTION DES COLONNES IMPORTANTES")
        print("=" * 60)
        
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
        # Trouver le dossier racine du projet
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Créer le dossier data/processed s'il n'existe pas
        data_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(data_dir, exist_ok=True)
        
        # Générer un nom de fichier si non fourni
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"players_processed_{timestamp}.csv"
        
        filepath = os.path.join(data_dir, filename)
        
        # Sauvegarder en CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"\nDonnées sauvegardées : {filepath}")
        print(f"{len(df)} joueurs × {len(df.columns)} colonnes")
        
        return filepath


# FONCTIONS UTILITAIRES


def quick_preprocess(min_minutes: int = 60, save: bool = True) -> Optional[pd.DataFrame]:
    """
    Fonction rapide pour prétraiter COMPLÈTEMENT les données.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLET DES DONNÉES FPL")
    print("=" * 60)
    
    preprocessor = FPLDataPreprocessor(min_minutes=min_minutes)
    
    # 1. Charger les données brutes
    raw_data = preprocessor.load_raw_data()
    if raw_data is None:
        return None
    
    # 2. Créer le DataFrame de base
    df = preprocessor.create_base_dataframe(raw_data)
    
    # 3. Filtrer les joueurs actifs
    df = preprocessor.filter_active_players(df)
    
    # 4. Ajouter les noms (positions et équipes)
    df = preprocessor.add_position_names(df, raw_data)
    df = preprocessor.add_team_names(df, raw_data)
    
    # 5. Feature Engineering - Performances
    df = preprocessor.engineer_performance_features(df)
    
    # 6. Feature Engineering - Fixtures
    df = preprocessor.engineer_fixture_features(df, raw_data)
    
    # 7. Sélectionner les colonnes importantes
    df = preprocessor.select_important_columns(df)
    
    # 8. Afficher un échantillon
    preprocessor.show_sample(df, n=10)
    
    # 9. Sauvegarder si demandé
    if save:
        print("\n" + "=" * 60)
        print("SAUVEGARDE DES DONNÉES")
        print("=" * 60)
        preprocessor.save_processed_data(df, filename="players_cleaned.csv")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    
    return df