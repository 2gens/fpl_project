"""
Tests avec pytest pour vérifier que les modèles fonctionnent correctement.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import pickle

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import FPLModelTrainer, train_all_models


class TestFPLModelTrainer:
    """Tests pour la classe FPLModelTrainer"""
    
    @pytest.fixture
    def trainer(self):
        """Fixture pour créer un trainer"""
        return FPLModelTrainer(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Fixture pour créer des données de test"""
        # Créer un DataFrame simple pour tester
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'id': range(n_samples),
            'web_name': [f'Player_{i}' for i in range(n_samples)],
            'team_name': ['Team_A'] * 50 + ['Team_B'] * 50,
            'position': ['MID'] * 40 + ['FWD'] * 30 + ['DEF'] * 20 + ['GKP'] * 10,
            'minutes': np.random.randint(60, 1000, n_samples),
            'form': np.random.uniform(2, 10, n_samples),
            'goals_per_90': np.random.uniform(0, 2, n_samples),
            'assists_per_90': np.random.uniform(0, 1, n_samples),
            'ict_index': np.random.uniform(20, 150, n_samples),
            'influence': np.random.uniform(10, 100, n_samples),
            'creativity': np.random.uniform(10, 100, n_samples),
            'threat': np.random.uniform(10, 100, n_samples),
            'next_fixture_difficulty': np.random.randint(1, 6, n_samples),
            'avg_fixture_difficulty_5': np.random.uniform(2, 4, n_samples),
            'price': np.random.uniform(4, 15, n_samples),
            'selected_by_percent': np.random.uniform(0, 50, n_samples),
            'goals_scored': np.random.randint(0, 20, n_samples),
            'assists': np.random.randint(0, 15, n_samples),
            'clean_sheets': np.random.randint(0, 10, n_samples),
            'bonus': np.random.randint(0, 20, n_samples),
            'yellow_cards': np.random.randint(0, 5, n_samples),
            'red_cards': np.random.randint(0, 2, n_samples),
            'penalties_order': np.random.randint(0, 2, n_samples),
            'corners_and_indirect_freekicks_order': np.random.randint(0, 2, n_samples),
            'total_points': np.random.randint(10, 150, n_samples)
        })
        
        return df
    
    def test_trainer_initialization(self, trainer):
        """Test que le trainer s'initialise correctement"""
        assert trainer is not None
        assert trainer.random_state == 42
        assert trainer.models == {}
        assert trainer.results == {}
    
    def test_load_processed_data(self, trainer):
        """Test le chargement des données"""
        df = trainer.load_processed_data()
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_select_features(self, trainer, sample_data):
        """Test la sélection des features"""
        X, y = trainer.select_features(sample_data)
        
        assert X is not None
        assert y is not None
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) == 100  # Notre sample a 100 joueurs
        
        # Vérifier que les features importantes sont présentes
        important_features = ['form', 'goals_per_90', 'assists_per_90', 'ict_index']
        for feature in important_features:
            assert feature in X.columns
    
    def test_split_data(self, trainer, sample_data):
        """Test la séparation train/test"""
        X, y = trainer.select_features(sample_data)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        assert len(X_train) == 80  # 80% de 100
        assert len(X_test) == 20   # 20% de 100
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_scale_features(self, trainer, sample_data):
        """Test la normalisation des features"""
        X, y = trainer.select_features(sample_data)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        assert X_train_scaled is not None
        assert X_test_scaled is not None
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
    
    def test_train_linear_regression(self, trainer, sample_data):
        """Test l'entraînement du modèle Linear Regression"""
        X, y = trainer.select_features(sample_data)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        X_train_scaled, _ = trainer.scale_features(X_train, X_test)
        
        model = trainer.train_linear_regression(X_train_scaled, y_train)
        
        assert model is not None
        assert 'Linear Regression' in trainer.models
        
        # Tester une prédiction
        predictions = model.predict(X_train_scaled)
        assert len(predictions) == len(y_train)
    
    def test_train_ridge_regression(self, trainer, sample_data):
        """Test l'entraînement du modèle Ridge Regression"""
        X, y = trainer.select_features(sample_data)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        X_train_scaled, _ = trainer.scale_features(X_train, X_test)
        
        model = trainer.train_ridge_regression(X_train_scaled, y_train, alpha=1.0)
        
        assert model is not None
        assert 'Ridge Regression' in trainer.models
    
    def test_train_random_forest(self, trainer, sample_data):
        """Test l'entraînement du modèle Random Forest"""
        X, y = trainer.select_features(sample_data)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        model = trainer.train_random_forest(X_train, y_train, n_estimators=10)
        
        assert model is not None
        assert 'Random Forest' in trainer.models
        assert hasattr(model, 'feature_importances_')
    
    def test_evaluate_model(self, trainer, sample_data):
        """Test l'évaluation d'un modèle"""
        X, y = trainer.select_features(sample_data)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        model = trainer.train_random_forest(X_train, y_train, n_estimators=10)
        results = trainer.evaluate_model(model, X_test, y_test, 'Random Forest')
        
        assert results is not None
        assert 'mae' in results
        assert 'rmse' in results
        assert 'r2' in results
        assert results['mae'] > 0
        assert results['rmse'] > 0
    
    def test_get_feature_importance(self, trainer, sample_data):
        """Test la récupération de l'importance des features"""
        X, y = trainer.select_features(sample_data)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        model = trainer.train_random_forest(X_train, y_train, n_estimators=10)
        importance_df = trainer.get_feature_importance('Random Forest', top_n=5)
        
        assert importance_df is not None
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns


class TestModelsExist:
    """Tests pour vérifier que les modèles sauvegardés existent"""
    
    def test_models_directory_exists(self):
        """Test que le dossier models existe"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "results", "models")
        
        assert os.path.exists(models_dir), "Le dossier results/models n'existe pas"
    
    def test_xgboost_model_exists(self):
        """Test que le modèle XGBoost existe"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "results", "models", "xgboost.pkl")
        
        assert os.path.exists(model_path), "Le modèle xgboost.pkl n'existe pas"
    
    def test_random_forest_model_exists(self):
        """Test que le modèle Random Forest existe"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "results", "models", "random_forest.pkl")
        
        assert os.path.exists(model_path), "Le modèle random_forest.pkl n'existe pas"
    
    def test_model_can_be_loaded(self):
        """Test qu'un modèle peut être chargé"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "results", "models", "xgboost.pkl")
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            assert model is not None
    
    def test_results_summary_exists(self):
        """Test que le fichier de résultats existe"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        results_path = os.path.join(project_root, "results", "models", "results_summary.json")
        
        assert os.path.exists(results_path), "Le fichier results_summary.json n'existe pas"


class TestMetrics:
    """Tests pour vérifier les métriques"""
    
    def test_mae_calculation(self):
        """Test le calcul du MAE"""
        from sklearn.metrics import mean_absolute_error
        
        y_true = np.array([100, 50, 75, 120])
        y_pred = np.array([95, 55, 70, 125])
        
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAE = (5 + 5 + 5 + 5) / 4 = 5
        assert mae == 5.0
    
    def test_rmse_calculation(self):
        """Test le calcul du RMSE"""
        from sklearn.metrics import mean_squared_error
        
        y_true = np.array([100, 50, 75, 120])
        y_pred = np.array([95, 55, 70, 125])
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # RMSE = sqrt((25 + 25 + 25 + 25) / 4) = sqrt(25) = 5
        assert rmse == 5.0


# Fonction pour lancer les tests si ce fichier est exécuté directement
if __name__ == "__main__":
    pytest.main([__file__, "-v"])