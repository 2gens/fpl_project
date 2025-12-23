"""
FPL Predictor Module : 
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

    def __init__(self):
        print("Predictor FPL initialisé")
    
    
    def load_player_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Charge les données des joueurs après nettoyage.
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
            'minutes', 'form', 'goals_per_90', 'assists_per_90', "momentum",
            'ict_index', 'influence', 'creativity', 'threat',
            'next_fixture_difficulty', 'avg_fixture_difficulty_5',
            'price', 'selected_by_percent',
            'goals_scored', 'assists', 'clean_sheets', 'bonus',
            'yellow_cards', 'red_cards',
            'penalties_order', 'corners_and_indirect_freekicks_order','expected_points_per_90', 'xM_factor', 'recent_5_minutes_ratio', 'recent_3_minutes_ratio',
            'historical_minutes_ratio', 'smart_adjustment', 'next_fixture_home',

        ]
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

    
    def predict_points_formula(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formule pondérée
        """
        print("Prédiction avec formule pondérée")
   
        df_pred = df.copy()
    
        form_normalized = df_pred['form'].clip(0, 10)
    
        expected_normalized = df_pred['expected_points_per_90'].clip(0, 10)
    
        ppg_normalized = df_pred['points_per_game'].clip(0, 10)
    
        fixture_bonus = df_pred['next_fixture_difficulty'].apply(
            lambda x: 2.0 if x <= 2 else 0.5 if x == 3 else -1.0
        )
        momentum_bonus = df_pred['momentum'].clip(-3, 3)
    
        # Formule Finale 
        df_pred['predicted_points'] = (
            (0.35 * form_normalized) +
            (0.25 * expected_normalized) +
            (0.20 * ppg_normalized) +
            (0.10 * fixture_bonus) +
            (0.10 * momentum_bonus)
        )
    
        # Bonus pour top scorers 
        proven_multiplier = 1 + (
            (df_pred['total_points'] - df_pred['total_points'].median()).clip(lower=0) / 
            df_pred['total_points'].max() * 0.3
        )
    
        df_pred['predicted_points'] = df_pred['predicted_points'] * proven_multiplier
    
        df_pred['predicted_value'] = df_pred['predicted_points'] / df_pred['price']

        print(f"\nPrédictions effectuées pour {len(df_pred)} joueurs")
    
        return df_pred
    
    def predict_points_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
         Modèle de prédiction ML
        """
   
        print("Prédiction avec modèle ML")
  
        # 3 meilleurs modèles
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "results", "models")
    
        models_to_load = {
            'xgboost': 0.5,
            'random_forest': 0.3,
            'gradient_boosting': 0.2
        }
    
        X = self.prepare_features(df)
    
        predictions_weighted = np.zeros(len(X))
        models_loaded = 0
    
        print("\nChargement et prédiction des modèles ML :")
    
        for model_name, weight in models_to_load.items():
            model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
                predictions = model.predict(X)
                predictions_weighted += predictions * weight
                models_loaded += 1
            
                print(f"  - {model_name.capitalize()} : poids {weight*100:.0f}%")
            else:
                print(f"  - {model_name.capitalize()} : non trouvé")
    
        if models_loaded == 0:
            print("\nErreur : Aucun modèle trouvé ")
            return df
    
        print(f"\n{models_loaded} modèles utilisés")
    
        df_pred = df.copy()
    
        df_pred['predicted_points'] = predictions_weighted

        # Système de pénalité basé sur le momentum et xM
        print("\nApplication du système de pénalité momentum/xM...")
        
        df_pred['momentum_penalty'] = 0
        
        # Niveau 1 : Momentum négatif seul (pénalité modérée)
        mask_negative_momentum = df_pred['momentum'] < -0.2
        df_pred.loc[mask_negative_momentum, 'momentum_penalty'] = (
            df_pred.loc[mask_negative_momentum, 'momentum'] * 1.3
        )

        # Niveau 2 : Momentum négatif + moins de temps de jeu (pénalité sévère)
        minutes_decline = (
            df_pred['recent_3_minutes_ratio'] < (df_pred['recent_5_minutes_ratio'] - 0.15)
        )
        
        mask_decline_severe = (df_pred['momentum'] < -0.3) & minutes_decline
        df_pred.loc[mask_decline_severe, 'momentum_penalty'] = (
            df_pred.loc[mask_decline_severe, 'momentum'] * 3.0
        )
        
        # Appliquer la pénalité
        df_pred['predicted_points'] = (
            df_pred['predicted_points'] + df_pred['momentum_penalty']
        )
        
        # Stats
        nb_penalise_modere = mask_negative_momentum.sum()
        nb_penalise_severe = mask_decline_severe.sum()
        print(f"Pénalité modérée (×1.5) : {nb_penalise_modere} joueurs")
        print(f"Pénalité sévère (×3.5) : {nb_penalise_severe} joueurs")
    
        df_pred['predicted_value'] = df_pred['predicted_points'] / df_pred['price']
    
        print(f"\nPrédictions ML effectuées pour {len(df_pred)} joueurs")
    
        return df_pred
    
    def get_top_players(self, df_pred: pd.DataFrame, n: int = 20, 
                   sort_by: str = 'predicted_points') -> pd.DataFrame:
        """
        Récupèrer le top des joueurs recommandés 
        """
        print(f"\nFiltrage intelligent des joueurs...")
        print(f"Joueurs avant filtrage : {len(df_pred)}")
    
        # Filtrage des joueurs
        df_filtered = df_pred[
            (df_pred['form'] >= 3.0) &           
            (df_pred['momentum'] >= -1.5) &     
            (df_pred['minutes'] >= 200)          
        ].copy()
    
        print(f"Joueurs après filtrage : {len(df_filtered)}")
        print(f"Joueurs éliminés : {len(df_pred) - len(df_filtered)}")
        
    
        cols_to_show = [
            'web_name', 'team_name', 'position', 'price',
            'predicted_points', 'predicted_value', 'form', 'momentum',
            'next_fixture_opponent', 'next_fixture_difficulty'
        ]
    
        available_cols = [col for col in cols_to_show if col in df_filtered.columns]
    
        top_players = df_filtered.nlargest(n, sort_by)[available_cols].copy()
    
        return top_players
    
    
    def get_recommendations_by_position(self, df_pred: pd.DataFrame, 
                                       n_per_position: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Recommandations par position.
        """
        recommendations = {}
        
        positions = ['GK', 'DEF', 'MID', 'FWD']
        
        for position in positions:
            pos_players = df_pred[df_pred['position'] == position].copy()
            top_pos = self.get_top_players(pos_players, n=n_per_position, sort_by='predicted_points')
            recommendations[position] = top_pos
        
        return recommendations
    

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
    

# Fonction utilitaire

def quick_predict(model_name: str = 'XGBoost', top_n: int = 20) -> Optional[pd.DataFrame]:

    predictor = FPLPredictor(model_name=model_name)
    
    if not predictor.load_model():
        return None
    
    df = predictor.load_player_data()
    if df is None:
        return None
    
    df_pred = predictor.predict_points_formula(df)

    recommendations = predictor.get_recommendations_by_position(df_pred, n_per_position=5)
    
    print("Sauvegarde des prédictions...")
    predictor.save_predictions(df_pred, filename="latest_predictions.csv")
    
    print("Prédictions terminées avec succès.")
    
    
    return df_pred