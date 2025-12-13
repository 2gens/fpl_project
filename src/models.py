"""
FPL Machine Learning Models Trainer
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle


class FPLModelTrainer:
    
    def __init__(self, random_state: int = 42):

        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
        
        print(f"Random state : {random_state}")
    
    def load_processed_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Charge les données nettoyées.
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
        print(f"Données chargées : {len(df)} joueurs, {len(df.columns)} colonnes")
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sélectionne les features (X) et la target (y) pour le ML. 
        """
        print("Sélection des features et de la target...")

        feature_cols = [
            # Performance stats
            'minutes', 'form', 'goals_per_90', 'assists_per_90', "momentum",
            
            # Indices
            'ict_index', 'influence', 'creativity', 'threat',
            
            # Fixtures
            'next_fixture_difficulty', 'avg_fixture_difficulty_5',
            
            # Prix et sélection
            'price', 'selected_by_percent',
            
            # Stats de base
            'goals_scored', 'assists', 'clean_sheets', 'bonus',
            'yellow_cards', 'red_cards',
            
            # Set pieces
            'penalties_order', 'corners_and_indirect_freekicks_order',

            # Expected stats
            'expected_points_per_90',

            # Features avancées -> mode pro 
            'xM_factor', 'recent_5_minutes_ratio', 
            'historical_minutes_ratio', 'smart_adjustment', 'next_fixture_home',
        ]
        
        # Ajouter expected stats si disponibles
        if 'expected_goals' in df.columns:
            feature_cols.append('expected_goals')
        if 'expected_assists' in df.columns:
            feature_cols.append('expected_assists')
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Créer X (features) et y (target)
        X = df[available_features].copy()
        y = df['recent_performance_score'].copy()
        
        # Gérer les valeurs manquantes
        X = X.fillna(0)
        
        print(f"Features sélectionnées : {len(available_features)}")
        print(f"Features : {', '.join(available_features)}")
        print(f"\nTarget :recent_performance_score (forme récente + xG/xA + fiabilité) ")
        print(f"Nombre d'exemples : {len(X)}")
        
        self.feature_names = available_features
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Sépare les données en train/test sets.
        """
        print("Séparation des données en train/test sets...")
      
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Train set : {len(X_train)} exemples ({(1-test_size)*100:.0f}%)")
        print(f"Test set  : {len(X_test)} exemples ({test_size*100:.0f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        """
        Normalisation des features (standardisation).
        """
        print("\nNormalisation des features...")
        
        # Fit sur train, transform sur train et test
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # DataFrame pour garder les noms de colonnes
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print("Features normalisées")
        
        return X_train_scaled, X_test_scaled
    
    def train_linear_regression(self, X_train, y_train) -> LinearRegression:
        """
        Entraîne le modèle Linear Regression (baseline).
        """
        
        print("Entrainement : Linear Regression")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        print("Modèle Linear Regression entraîné")
        self.models['Linear Regression'] = model
        
        return model
    
    def train_ridge_regression(self, X_train, y_train, alpha: float = 1.0) -> Ridge:
        """
        Entraîne le modèle Ridge Regression (régularisation L2).
        """
    
        print("Entrainement : Ridge Regression")
     
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        print(f"Modèle Ridge Regression entraîné (alpha={alpha})")
        self.models['Ridge Regression'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators: int = 100) -> RandomForestRegressor:
        """
        Entraîne le modèle Random Forest.
        """
        print("Entrainement : Random Forest")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1  
        )
        model.fit(X_train, y_train)
        
        print(f"Modèle Random Forest entraîné ({n_estimators} arbres)")
        self.models['Random Forest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators: int = 100) -> xgb.XGBRegressor:
        """
        Entraîne le modèle XGBoost.
        """
        print("Entrainement : XGBoost")
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        print(f"Modèle XGBoost entraîné ({n_estimators} arbres)")
        self.models['XGBoost'] = model
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, n_estimators: int = 100) -> GradientBoostingRegressor:
        """
        Entraîne le modèle Gradient Boosting.
        """
        print("Entrainement : Gradient Boosting")
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        
        print(f"Modèle Gradient Boosting entraîné ({n_estimators} arbres)")
        self.models['Gradient Boosting'] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """
        Évalue un modèle sur le test set.
        """
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Calculer les métriques
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Stocker les résultats
        results = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        self.results[model_name] = results
        
        return results
    
    def cross_validate_model(self, model, X, y, model_name: str, cv: int = 5) -> Dict:
        """
        Validation croisée pour tester la robustesse du modèle.
        """
        print(f"\nValidation croisée ({cv}-fold) pour {model_name}...")
        
        # MAE comme métrique
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae_scores = -scores  # Convertir en positif
        
        cv_results = {
            'cv_mae_mean': mae_scores.mean(),
            'cv_mae_std': mae_scores.std()
        }
        
        print(f"CV MAE : {cv_results['cv_mae_mean']:.2f} (+/- {cv_results['cv_mae_std']:.2f})")
        
        # Ajouter aux résultats existants
        if model_name in self.results:
            self.results[model_name].update(cv_results)
        
        return cv_results
    
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> Optional[pd.DataFrame]:
        """
        Récupère l'importance des features pour Random Forest, XGBoost, ou Gradient Boosting.
        """
        if model_name not in self.models:
            print(f"Erreur : Modèle {model_name} non trouvé")
            return None
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Le modèle {model_name} n'a pas de feature importance")
            return None
        
        # DataFrame des importances
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def print_results_summary(self):
        """
        Résumé comparatif de tous les modèles.
        """
        print("Résumé des performances des modèles")
        
        
        if not self.results:
            print("Aucun résultat disponible")
            return
        
        # DataFrame pour comparaison
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('mae')
        
        print("\nClassement par MAE (Mean Absolute Error) :")
        print(results_df[['mae', 'rmse', 'r2']].to_string())
        
        # Meilleur modèle
        best_model = results_df.index[0]
        best_mae = results_df.loc[best_model, 'mae']
        
        print(f"\nMeilleur modèle : {best_model}")
        print(f"MAE : {best_mae:.2f} points")
        print(f"Cela signifie une erreur moyenne de {best_mae:.2f} points par prédiction")
    
    def save_models(self, output_dir: str = None):
        """
        Sauvegarde tous les modèles entraînés.
        """
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            output_dir = os.path.join(project_root, "results", "models")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder chaque modèle
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Modèle sauvegardé : {filepath}")
        
        # Sauvegarder aussi les résultats
        results_file = os.path.join(output_dir, "results_summary.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Résultats sauvegardés : {results_file}")


# Fonction utilitaire

def train_all_models(min_minutes: int = 60, test_size: float = 0.2) -> FPLModelTrainer:

    trainer = FPLModelTrainer(random_state=42)
    
    df = trainer.load_processed_data()
    if df is None:
        return None
    
    X, y = trainer.select_features(df)
    
    # Split train/test
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=test_size)
    
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
    
    #Regression Linéaire
    lr_model = trainer.train_linear_regression(X_train_scaled, y_train)
    lr_results = trainer.evaluate_model(lr_model, X_test_scaled, y_test, 'Linear Regression')
    print(f"MAE : {lr_results['mae']:.2f}, RMSE : {lr_results['rmse']:.2f}, R² : {lr_results['r2']:.3f}")
    
    # Ridge Regression 
    ridge_model = trainer.train_ridge_regression(X_train_scaled, y_train, alpha=1.0)
    ridge_results = trainer.evaluate_model(ridge_model, X_test_scaled, y_test, 'Ridge Regression')
    print(f"MAE : {ridge_results['mae']:.2f}, RMSE : {ridge_results['rmse']:.2f}, R² : {ridge_results['r2']:.3f}")
    
    # Random Forest 
    rf_model = trainer.train_random_forest(X_train, y_train, n_estimators=100)
    rf_results = trainer.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    print(f"MAE : {rf_results['mae']:.2f}, RMSE : {rf_results['rmse']:.2f}, R² : {rf_results['r2']:.3f}")
    
    # XGBoost 
    xgb_model = trainer.train_xgboost(X_train, y_train, n_estimators=100)
    xgb_results = trainer.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    print(f"MAE : {xgb_results['mae']:.2f}, RMSE : {xgb_results['rmse']:.2f}, R² : {xgb_results['r2']:.3f}")
    
    # Gradient Boosting 
    gb_model = trainer.train_gradient_boosting(X_train, y_train, n_estimators=100)
    gb_results = trainer.evaluate_model(gb_model, X_test, y_test, 'Gradient Boosting')
    print(f"MAE : {gb_results['mae']:.2f}, RMSE : {gb_results['rmse']:.2f}, R² : {gb_results['r2']:.3f}")
    
    trainer.print_results_summary()
    
    
    for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
        print(f"\n{model_name} :")
        importance_df = trainer.get_feature_importance(model_name, top_n=10)
        if importance_df is not None:
            print(importance_df.to_string(index=False))
    
    print("Sauvegarde des modèles entraînés...")
    trainer.save_models()

    print("Entrainement terminé.")
    
    return trainer