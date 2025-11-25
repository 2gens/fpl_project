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
            'minutes', 'form', 'goals_per_90', 'assists_per_90', "momentum",
            'ict_index', 'influence', 'creativity', 'threat',
            'next_fixture_difficulty', 'avg_fixture_difficulty_5',
            'price', 'selected_by_percent',
            'goals_scored', 'assists', 'clean_sheets', 'bonus',
            'yellow_cards', 'red_cards',
            'penalties_order', 'corners_and_indirect_freekicks_order','expected_points_per_90',
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

        df_pred['predicted_points_raw'] = df_pred['predicted_points']

        # Ajustement basé sur forme et momentum
        print("\nApplication des ajustements forme/momentum...")
    
        high_momentum_boost = 1 + (df_pred['momentum'].clip(lower=0) * 0.15)
        low_momentum_penalty = df_pred['momentum'].apply(lambda x: 0.85 if x < -1 else 1.0)
    
        form_boost = 1 + ((df_pred['form'] - 5).clip(lower=0) * 0.08)
    
 
        df_pred['predicted_points'] = (
            df_pred['predicted_points_raw'] * 
            high_momentum_boost * 
            low_momentum_penalty * 
            form_boost
        )
    
        print("Ajustements appliqués :")
        print("  - Boost momentum : +15% par point positif")
        print("  - Boost form : +8% par point au-dessus de 5")
        print("  - Pénalité baisse : -15% si momentum < -1")

        # Calculer le "value" prédit (predicted_points / price)
        df_pred['predicted_value'] = df_pred['predicted_points'] / df_pred['price']
        
        print(f"Prédictions effectuées pour {len(df_pred)} joueurs")
        
        return df_pred
    
    def predict_points_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les points en utilisant un ENSEMBLE de modèles.
        Plus robuste et précis qu'un seul modèle !
        """
        print("\n" + "=" * 60)
        print("PRÉDICTION AVEC ENSEMBLE DE MODÈLES")
        print("=" * 60)
    
        # Charger les 3 meilleurs modèles
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "results", "models")
    
        # Poids de chaque modèle (total = 1.0)
        models_to_load = {
            'xgboost': 0.5,           
            'random_forest': 0.3,     
            'gradient_boosting': 0.2  
        }
        # Préparer les features
        X = self.prepare_features(df)
    
        # Stocker les prédictions pondérées
        predictions_weighted = np.zeros(len(X))
        models_loaded = 0
    
        print("\nChargement et prédiction des modèles :")
    
        for model_name, weight in models_to_load.items():
            model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
                predictions = model.predict(X)
                predictions_weighted += predictions * weight
                models_loaded += 1
            
                print(f"  - {model_name.capitalize()} : poids {weight*100:.0f}% ")
            else:
                print(f"  - {model_name.capitalize()} : non trouvé ")
    
        if models_loaded == 0:
            print("\nErreur : Aucun modèle trouvé !")
            return df
    
        print(f"\n{models_loaded} modèles utilisés pour l'ensemble")

        # Créer le DataFrame de prédiction
        df_pred = df.copy()
        df_pred['predicted_points_raw'] = predictions_weighted
    
        # Le modèle prédit les points TOTAUX, pas par gameweek. On divise par le nombre de matchs pour avoir une prédiction réaliste
        df_pred['appearances'] = (df_pred['minutes'] / 90).clip(lower=1)
        df_pred['predicted_points_normalized'] = df_pred['predicted_points_raw'] / df_pred['appearances']
    
        print("\nNormalisation des prédictions (points par match)...")
    
        # Ajustements basé sur forme et momentum
        print("Application des ajustements forme/momentum...")
    
        high_momentum_boost = 1 + (df_pred['momentum'].clip(lower=0) * 0.15)
        low_momentum_penalty = df_pred.apply(
            lambda row: 0.95 if row['momentum'] < -1 and row['total_points'] >= 100 
                        else 0.85 if row['momentum'] < -1 
                        else 1.0,
            axis=1
        )
    
        form_boost = 1 + ((df_pred['form'] - 5).clip(lower=0) * 0.08)
    
        median_points = df_pred['total_points'].median()
        max_points = df_pred['total_points'].max()
        proven_boost = 1 + (
            (df_pred['total_points'] - median_points).clip(lower=0) / 
            max_points * 0.25  # +25% max pour les top scorers
        )
    
        # Appliquer tous les ajustements
        df_pred['predicted_points'] = (
            df_pred['predicted_points_normalized'] * 
            high_momentum_boost * 
            low_momentum_penalty * 
            form_boost *
            proven_boost
        )
    
        print("Ajustements appliqués :")
        print("  - Normalisation par nombre de matchs")
        print("  - Boost momentum : +15% par point positif")
        print("  - Boost form : +8% par point au-dessus de 5")
        print("  - Boost proven players : +25% max pour top scorers")
        print("  - Pénalité baisse adaptative")
    
        # Calculer le "value" prédit
        df_pred['predicted_value'] = df_pred['predicted_points'] / df_pred['price']
    
        print(f"\nPrédictions ensemble effectuées pour {len(df_pred)} joueurs")
    
        return df_pred
    
    
    def get_top_players(self, df_pred: pd.DataFrame, n: int = 20, 
                   sort_by: str = 'predicted_points') -> pd.DataFrame:
        """
        Récupère les top joueurs AVEC FILTRAGE INTELLIGENT.
        """
        print(f"\nFiltrage intelligent des joueurs...")
        print(f"Joueurs avant filtrage : {len(df_pred)}")
    
        # FILTRAGE INTELLIGENT
        df_filtered = df_pred[
            (df_pred['form'] >= 3.0) &           # Forme récente correcte
            (df_pred['momentum'] >= -1.5) &      # Pas en chute libre
            (df_pred['minutes'] >= 200)          # A joué suffisamment
        ].copy()
    
        print(f"Joueurs après filtrage : {len(df_filtered)}")
        print(f"Joueurs éliminés : {len(df_pred) - len(df_filtered)}")
        print("\nCritères de filtrage appliqués :")
        print("  - Form >= 3.0 (forme récente correcte)")
        print("  - Momentum >= -1.5 (pas en chute libre)")
        print("  - Minutes >= 200 (temps de jeu suffisant)")
    
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
    
        if len(top) == 0:
            print("\nAucun joueur ne correspond aux critères de filtrage")
            return
    
        # Formater l'affichage
        print(f"\n{'Rang':<5} {'Joueur':<20} {'Équipe':<10} {'Pos':<4} {'Prix':<8} {'Pts prédits':<13} {'Form':<6} {'Mom':<6} {'Adversaire':<15}")
        print("-" * 115)
    
        for idx, (_, row) in enumerate(top.iterrows(), 1):
            joueur = row['web_name'][:18]
            equipe = row['team_name'][:8]
            position = row['position']
            prix = f"£{row['price']:.1f}M"
            pts_pred = f"{row['predicted_points']:.1f}"
            form = f"{row['form']:.1f}"
            momentum = f"{row['momentum']:.1f}"
            adversaire = row.get('next_fixture_opponent', 'N/A')[:13]
        
            # Marquer les joueurs en super forme
            marker = " 🔥" if row['momentum'] > 2.0 else ""
        
            print(f"{idx:<5} {joueur:<20} {equipe:<10} {position:<4} {prix:<8} {pts_pred:<13} {form:<6} {momentum:<6} {adversaire:<15}{marker}")
    
        print()
    
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
    df_pred = predictor.predict_points_ensemble(df)
    
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