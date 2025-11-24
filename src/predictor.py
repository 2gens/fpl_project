"""
FPL Predictor Module
Ce module utilise les modèles entraînés pour faire des prédictions
et générer des recommandations de joueurs pour la prochaine gameweek.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class FPLPredictor:
    """
    Classe pour faire des prédictions FPL avec les modèles entraînés.
    """
    
    def __init__(self, model_name: str = 'XGBoost'):
        """
        Initialise le prédictor.
        """
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        print(f"Predictor initialisé avec le modèle : {model_name}")
    
    def load_model(self, model_path: str = None):
        """
        Charge un modèle sauvegardé.
        """
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_filename = f"{self.model_name.lower().replace(' ', '_')}.pkl"
            model_path = os.path.join(project_root, "results", "models", model_filename)
        
        if not os.path.exists(model_path):
            print(f"Erreur : Modèle non trouvé - {model_path}")
            return False
        
        print(f"Chargement du modèle depuis : {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print("Modèle chargé avec succès")
        return True
    
    def load_player_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Charge les données des joueurs preprocessées.
        """
        if filepath is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            filepath = os.path.join(project_root, "data", "processed", "players_cleaned.csv")
        
        if not os.path.exists(filepath):
            print(f"Erreur : Fichier non trouvé - {filepath}")
            return None
        
        print(f"Chargement des données depuis : {filepath}")
        df = pd.read_csv(filepath)
        print(f"Données chargées : {len(df)} joueurs")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les features pour la prédiction. Utilise les mêmes features que lors de l'entraînement. 
        """
        feature_cols = [
            'minutes', 'form', 'goals_per_90', 'assists_per_90',
            'ict_index', 'influence', 'creativity', 'threat',
            'next_fixture_difficulty', 'avg_fixture_difficulty_5',
            'price', 'selected_by_percent',
            'goals_scored', 'assists', 'clean_sheets', 'bonus',
            'yellow_cards', 'red_cards',
            'penalties_order', 'corners_and_indirect_freekicks_order',
        ]
        
        # Ajouter expected stats si disponibles
        if 'expected_goals' in df.columns:
            feature_cols.append('expected_goals')
        if 'expected_assists' in df.columns:
            feature_cols.append('expected_assists')
        
        # Garder seulement les colonnes qui existent
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].copy()
        X = X.fillna(0)
        
        self.feature_names = available_features
        
        return X
    
    def predict_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les points pour tous les joueurs.
        """
        if self.model is None:
            print("Erreur : Aucun modèle chargé")
            return df
        
        print("\n" + "=" * 60)
        print("PRÉDICTION DES POINTS")
        print("=" * 60)
        
        # Préparer les features
        X = self.prepare_features(df)
        
        # Faire les prédictions
        predictions = self.model.predict(X)
        
        # Ajouter les prédictions au DataFrame
        df_pred = df.copy()
        df_pred['predicted_points'] = predictions
        
        # Calculer le "value" prédit (predicted_points / price)
        df_pred['predicted_value'] = df_pred['predicted_points'] / df_pred['price']
        
        print(f"Prédictions effectuées pour {len(df_pred)} joueurs")
        
        return df_pred
    
    def get_top_players(self, df_pred: pd.DataFrame, n: int = 20, 
                       sort_by: str = 'predicted_points') -> pd.DataFrame:
        """
        Récupère les top joueurs selon un critère
        """
        cols_to_show = [
            'web_name', 'team_name', 'position', 'price',
            'predicted_points', 'predicted_value',
            'next_fixture_opponent', 'next_fixture_difficulty'
        ]
        
        available_cols = [col for col in cols_to_show if col in df_pred.columns]
        
        top_players = df_pred.nlargest(n, sort_by)[available_cols].copy()
        
        return top_players
    
    def get_recommendations_by_position(self, df_pred: pd.DataFrame, 
                                       n_per_position: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Génère des recommandations par position.
        """
        recommendations = {}
        
        positions = ['GK', 'DEF', 'MID', 'FWD']
        
        for position in positions:
            pos_players = df_pred[df_pred['position'] == position].copy()
            top_pos = self.get_top_players(pos_players, n=n_per_position, sort_by='predicted_points')
            recommendations[position] = top_pos
        
        return recommendations
    
    def print_top_players(self, df_pred: pd.DataFrame, n: int = 20):
        """
        Affiche les top joueurs recommandés.
        """
        print("\n" + "=" * 60)
        print(f"TOP {n} JOUEURS RECOMMANDÉS (par points prédits)")
        print("=" * 60)
        
        top = self.get_top_players(df_pred, n=n, sort_by='predicted_points')
        
        # Formater l'affichage
        print(f"\n{'Rang':<5} {'Joueur':<20} {'Équipe':<8} {'Pos':<4} {'Prix':<7} {'Pts prédits':<12} {'Value':<8} {'Adversaire':<15}")
        print("-" * 100)
        
        for idx, (_, row) in enumerate(top.iterrows(), 1):
            joueur = row['web_name'][:18]
            equipe = row['team_name'][:6]
            position = row['position']
            prix = f"£{row['price']:.1f}M"
            pts_pred = f"{row['predicted_points']:.1f}"
            value = f"{row['predicted_value']:.2f}"
            adversaire = row.get('next_fixture_opponent', 'N/A')[:13]
            
            # Marquer les meilleurs "value"
            marker = " ⭐" if row['predicted_value'] > 0.7 else ""
            
            print(f"{idx:<5} {joueur:<20} {equipe:<8} {position:<4} {prix:<7} {pts_pred:<12} {value:<8} {adversaire:<15}{marker}")
    
    def print_recommendations_by_position(self, recommendations: Dict[str, pd.DataFrame]):
        """
        Affiche les recommandations par position.
        """
        print("\n" + "=" * 60)
        print("RECOMMANDATIONS PAR POSITION")
        print("=" * 60)
        
        position_names = {
            'GK': 'GARDIENS',
            'DEF': 'DÉFENSEURS',
            'MID': 'MILIEUX',
            'FWD': 'ATTAQUANTS'
        }
        
        for position, df_pos in recommendations.items():
            print(f"\n{position_names[position]} (Top 5) :")
            print("-" * 80)
            
            for idx, (_, row) in enumerate(df_pos.iterrows(), 1):
                joueur = row['web_name']
                equipe = row['team_name']
                prix = row['price']
                pts_pred = row['predicted_points']
                value = row['predicted_value']
                
                print(f"  {idx}. {joueur} ({equipe}) - £{prix:.1f}M - {pts_pred:.1f} pts - Value: {value:.2f}")
    
    def save_predictions(self, df_pred: pd.DataFrame, filename: str = None) -> str:
        """
        Sauvegarde les prédictions en CSV.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        output_dir = os.path.join(project_root, "data", "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"predictions_{timestamp}.csv"
        
        filepath = os.path.join(output_dir, filename)
        
        # Colonnes à sauvegarder
        cols_to_save = [
            'id', 'web_name', 'team_name', 'position', 'price',
            'total_points', 'predicted_points', 'predicted_value',
            'form', 'minutes', 'goals_per_90', 'assists_per_90',
            'next_fixture_opponent', 'next_fixture_difficulty', 'avg_fixture_difficulty_5'
        ]
        
        available_cols = [col for col in cols_to_save if col in df_pred.columns]
        
        df_to_save = df_pred[available_cols].copy()
        df_to_save = df_to_save.sort_values('predicted_points', ascending=False)
        
        df_to_save.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"\nPrédictions sauvegardées : {filepath}")
        print(f"{len(df_to_save)} joueurs classés par points prédits")
        
        return filepath

# FONCTIONS UTILITAIRES

def quick_predict(model_name: str = 'XGBoost', top_n: int = 20) -> Optional[pd.DataFrame]:
    """
    Fonction rapide pour faire des prédictions.
    
    Args:
        model_name (str): Nom du modèle à utiliser
        top_n (int): Nombre de top joueurs à afficher
    
    Returns:
        pd.DataFrame: DataFrame avec prédictions
    """
    print("\n" + "=" * 60)
    print("PRÉDICTIONS FPL POUR LA PROCHAINE GAMEWEEK")
    print("=" * 60)
    
    # 1. Initialiser le predictor
    predictor = FPLPredictor(model_name=model_name)
    
    # 2. Charger le modèle
    if not predictor.load_model():
        return None
    
    # 3. Charger les données
    df = predictor.load_player_data()
    if df is None:
        return None
    
    # 4. Faire les prédictions
    df_pred = predictor.predict_points(df)
    
    # 5. Afficher les top joueurs
    predictor.print_top_players(df_pred, n=top_n)
    
    # 6. Afficher les recommandations par position
    recommendations = predictor.get_recommendations_by_position(df_pred, n_per_position=5)
    predictor.print_recommendations_by_position(recommendations)
    
    # 7. Sauvegarder les prédictions
    print("\n" + "=" * 60)
    print("SAUVEGARDE DES PRÉDICTIONS")
    print("=" * 60)
    predictor.save_predictions(df_pred, filename="latest_predictions.csv")
    
    print("\n" + "=" * 60)
    print("PRÉDICTIONS TERMINÉES AVEC SUCCÈS")
    print("=" * 60)
    
    return df_pred