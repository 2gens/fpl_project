"""
FPL Machine Learning Models Module
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
    """
    Classe pour entraîner et évaluer les modèles de prédiction FPL.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise le trainer.
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
        
        print("Model Trainer initialisé")
        print(f"Random state : {random_state}")
    
    def load_processed_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Charge les données preprocessées depuis le CSV.
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
        Sélectionne les features (X) et la target (y) pour le ML. Target = total_points (ce qu'on veut prédire)
        """
        print("\n" + "=" * 60)
        print("SÉLECTION DES FEATURES POUR LE ML")
        print("=" * 60)
        
        # Features à utiliser pour la prédiction
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
        ]
        
        # Ajouter expected stats si disponibles
        if 'expected_goals' in df.columns:
            feature_cols.append('expected_goals')
        if 'expected_assists' in df.columns:
            feature_cols.append('expected_assists')
        
        # Garder seulement les colonnes qui existent
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Créer X (features) et y (target)
        X = df[available_features].copy()
        y = df['total_points'].copy()
        
        # Gérer les valeurs manquantes
        X = X.fillna(0)
        
        print(f"Features sélectionnées : {len(available_features)}")
        print(f"Features : {', '.join(available_features)}")
        print(f"\nTarget : total_points")
        print(f"Nombre d'exemples : {len(X)}")
        
        self.feature_names = available_features
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Sépare les données en train/test sets.
        """
        print("\n" + "=" * 60)
        print("SÉPARATION TRAIN/TEST")
        print("=" * 60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Train set : {len(X_train)} exemples ({(1-test_size)*100:.0f}%)")
        print(f"Test set  : {len(X_test)} exemples ({test_size*100:.0f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        """
        Normalise les features (standardisation).
        """
        print("\nNormalisation des features...")
        
        # Fit sur train, transform sur train et test
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Reconvertir en DataFrame pour garder les noms de colonnes
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print("Features normalisées")
        
        return X_train_scaled, X_test_scaled
    
    def train_linear_regression(self, X_train, y_train) -> LinearRegression:
        """
        Entraîne le modèle Linear Regression (baseline).
        """
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT : LINEAR REGRESSION")
        print("=" * 60)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        print("Modèle Linear Regression entraîné")
        self.models['Linear Regression'] = model
        
        return model
    
    def train_ridge_regression(self, X_train, y_train, alpha: float = 1.0) -> Ridge:
        """
        Entraîne le modèle Ridge Regression (régularisation L2).
        """
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT : RIDGE REGRESSION")
        print("=" * 60)
        
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        print(f"Modèle Ridge Regression entraîné (alpha={alpha})")
        self.models['Ridge Regression'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators: int = 100) -> RandomForestRegressor:
        """
        Entraîne le modèle Random Forest.
        """
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT : RANDOM FOREST")
        print("=" * 60)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1  # Utiliser tous les CPU
        )
        model.fit(X_train, y_train)
        
        print(f"Modèle Random Forest entraîné ({n_estimators} arbres)")
        self.models['Random Forest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators: int = 100) -> xgb.XGBRegressor:
        """
        Entraîne le modèle XGBoost.
        """
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT : XGBOOST")
        print("=" * 60)
        
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
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT : GRADIENT BOOSTING")
        print("=" * 60)
        
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
        Effectue une validation croisée pour tester la robustesse du modèle.
        """
        print(f"\nValidation croisée ({cv}-fold) pour {model_name}...")
        
        # Scoring avec MAE (négatif dans sklearn, donc on prend l'opposé)
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
        
        # Vérifier si le modèle a feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"Le modèle {model_name} n'a pas de feature importance")
            return None
        
        # Créer DataFrame avec importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def print_results_summary(self):
        """
        Affiche un résumé comparatif de tous les modèles.
        """
        print("\n" + "=" * 60)
        print("RÉSUMÉ DES RÉSULTATS - COMPARAISON DES MODÈLES")
        print("=" * 60)
        
        if not self.results:
            print("Aucun résultat disponible")
            return
        
        # Créer DataFrame pour comparaison
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


# FONCTIONS UTILITAIRES

def train_all_models(min_minutes: int = 60, test_size: float = 0.2) -> FPLModelTrainer:
    """
    Fonction complète pour entraîner tous les modèles.
    """
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT COMPLET DES MODÈLES FPL")
    print("=" * 60)
    
    # 1. Initialiser le trainer
    trainer = FPLModelTrainer(random_state=42)
    
    # 2. Charger les données
    df = trainer.load_processed_data()
    if df is None:
        return None
    
    # 3. Sélectionner features et target
    X, y = trainer.select_features(df)
    
    # 4. Split train/test
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=test_size)
    
    # 5. Normaliser les features
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
    
    # 6. Entraîner tous les modèles
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT DES 5 MODÈLES")
    print("=" * 60)
    
    # Linear Regression (avec features normalisées)
    lr_model = trainer.train_linear_regression(X_train_scaled, y_train)
    lr_results = trainer.evaluate_model(lr_model, X_test_scaled, y_test, 'Linear Regression')
    print(f"MAE : {lr_results['mae']:.2f}, RMSE : {lr_results['rmse']:.2f}, R² : {lr_results['r2']:.3f}")
    
    # Ridge Regression (avec features normalisées)
    ridge_model = trainer.train_ridge_regression(X_train_scaled, y_train, alpha=1.0)
    ridge_results = trainer.evaluate_model(ridge_model, X_test_scaled, y_test, 'Ridge Regression')
    print(f"MAE : {ridge_results['mae']:.2f}, RMSE : {ridge_results['rmse']:.2f}, R² : {ridge_results['r2']:.3f}")
    
    # Random Forest (features non normalisées, meilleur pour les arbres)
    rf_model = trainer.train_random_forest(X_train, y_train, n_estimators=100)
    rf_results = trainer.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    print(f"MAE : {rf_results['mae']:.2f}, RMSE : {rf_results['rmse']:.2f}, R² : {rf_results['r2']:.3f}")
    
    # XGBoost (features non normalisées)
    xgb_model = trainer.train_xgboost(X_train, y_train, n_estimators=100)
    xgb_results = trainer.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    print(f"MAE : {xgb_results['mae']:.2f}, RMSE : {xgb_results['rmse']:.2f}, R² : {xgb_results['r2']:.3f}")
    
    # Gradient Boosting (features non normalisées)
    gb_model = trainer.train_gradient_boosting(X_train, y_train, n_estimators=100)
    gb_results = trainer.evaluate_model(gb_model, X_test, y_test, 'Gradient Boosting')
    print(f"MAE : {gb_results['mae']:.2f}, RMSE : {gb_results['rmse']:.2f}, R² : {gb_results['r2']:.3f}")
    
    # 7. Afficher le résumé
    trainer.print_results_summary()
    
    # 8. Feature importance pour les meilleurs modèles
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (TOP 10)")
    print("=" * 60)
    
    for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
        print(f"\n{model_name} :")
        importance_df = trainer.get_feature_importance(model_name, top_n=10)
        if importance_df is not None:
            print(importance_df.to_string(index=False))
    
    # 9. Sauvegarder les modèles
    print("\n" + "=" * 60)
    print("SAUVEGARDE DES MODÈLES")
    print("=" * 60)
    trainer.save_models()
    
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    
    return trainer