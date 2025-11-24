"""
Tests avec pytest pour vérifier que le preprocessing fonctionne correctement.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import FPLDataPreprocessor, quick_preprocess


class TestFPLDataPreprocessor:
    """Tests pour la classe FPLDataPreprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        """Fixture pour créer un preprocessor"""
        return FPLDataPreprocessor(min_minutes=60)
    
    @pytest.fixture
    def sample_data(self):
        """Fixture pour créer des données de test"""
        return {
            'bootstrap': {
                'elements': [
                   #Exemple de joueurs ... = pas des vrais données  
                    {
                        'id': 1, 'web_name': 'Salah', 'team': 10, 'element_type': 3,
                        'now_cost': 130, 'minutes': 900, 'total_points': 100,
                        'goals_scored': 10, 'assists': 5, 'form': '8.0',
                        'ict_index': '120.0', 'clean_sheets': 3
                    },
                    {
                        'id': 2, 'web_name': 'Haaland', 'team': 11, 'element_type': 4,
                        'now_cost': 150, 'minutes': 800, 'total_points': 120,
                        'goals_scored': 15, 'assists': 2, 'form': '9.0',
                        'ict_index': '130.0', 'clean_sheets': 1
                    },
                    {
                        'id': 3, 'web_name': 'Bench Player', 'team': 5, 'element_type': 2,
                        'now_cost': 40, 'minutes': 20, 'total_points': 5,
                        'goals_scored': 0, 'assists': 0, 'form': '1.0',
                        'ict_index': '10.0', 'clean_sheets': 0
                    }
                ],
                'teams': [
                    {'id': 10, 'name': 'Liverpool'},
                    {'id': 11, 'name': 'Man City'},
                    {'id': 5, 'name': 'Everton'}
                ],
                'element_types': [
                    {'id': 1, 'singular_name_short': 'GKP'},
                    {'id': 2, 'singular_name_short': 'DEF'},
                    {'id': 3, 'singular_name_short': 'MID'},
                    {'id': 4, 'singular_name_short': 'FWD'}
                ]
            }
        }
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test que le preprocessor s'initialise correctement"""
        assert preprocessor is not None
        assert preprocessor.min_minutes == 60
    
    def test_create_base_dataframe(self, preprocessor, sample_data):
        """Test la création du DataFrame de base"""
        df = preprocessor.create_base_dataframe(sample_data)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'web_name' in df.columns
        assert 'total_points' in df.columns
    
    def test_filter_active_players(self, preprocessor, sample_data):
        """Test le filtrage des joueurs actifs"""
        df = preprocessor.create_base_dataframe(sample_data)
        df_filtered = preprocessor.filter_active_players(df)
        
        # Avec min_minutes=60, seuls 2 joueurs doivent rester (Salah et Haaland)
        assert len(df_filtered) == 2
        assert 'Salah' in df_filtered['web_name'].values
        assert 'Haaland' in df_filtered['web_name'].values
        assert 'Bench Player' not in df_filtered['web_name'].values
    
    def test_add_position_names(self, preprocessor, sample_data):
        """Test l'ajout des noms de positions"""
        df = preprocessor.create_base_dataframe(sample_data)
        df = preprocessor.add_position_names(df, sample_data)
        
        assert 'position' in df.columns
        assert 'MID' in df['position'].values
        assert 'FWD' in df['position'].values
    
    def test_add_team_names(self, preprocessor, sample_data):
        """Test l'ajout des noms d'équipes"""
        df = preprocessor.create_base_dataframe(sample_data)
        df = preprocessor.add_team_names(df, sample_data)
        
        assert 'team_name' in df.columns
        assert 'Liverpool' in df['team_name'].values
        assert 'Man City' in df['team_name'].values
    
    def test_engineer_performance_features(self, preprocessor, sample_data):
        """Test le feature engineering des performances"""
        df = preprocessor.create_base_dataframe(sample_data)
        df = preprocessor.engineer_performance_features(df)
        
        # Vérifier que les nouvelles colonnes existent
        assert 'price' in df.columns
        assert 'goals_per_90' in df.columns
        assert 'assists_per_90' in df.columns
        assert 'points_per_game' in df.columns
        assert 'points_per_million' in df.columns
        
        # Vérifier les calcul. Par ex, Salah : 130 / 10 = 13.0
        salah = df[df['web_name'] == 'Salah'].iloc[0]
        assert salah['price'] == 13.0
        
        # Salah : (10 goals * 90) / 900 minutes = 1.0 goals_per_90
        assert salah['goals_per_90'] == 1.0
        
        # Salah : 100 points / 13.0 price = 7.69 points_per_million
        assert abs(salah['points_per_million'] - 7.69) < 0.01
    
    def test_goals_per_90_calculation(self, preprocessor):
        """Test spécifique du calcul goals_per_90"""
        # Créer un DataFrame simple
        df = pd.DataFrame({
            'goals_scored': [10, 5, 0],
            'minutes': [900, 450, 0]
        })
        
        df['goals_per_90'] = np.where(
            df['minutes'] > 0,
            (df['goals_scored'] * 90) / df['minutes'],
            0
        )
        
        # Vérifier les calculs
        assert df.iloc[0]['goals_per_90'] == 1.0  # (10 * 90) / 900 = 1.0
        assert df.iloc[1]['goals_per_90'] == 1.0  # (5 * 90) / 450 = 1.0
        assert df.iloc[2]['goals_per_90'] == 0.0  # Division par 0 évitée


class TestPreprocessingPipeline:
    """Tests pour le pipeline complet"""
    
    def test_quick_preprocess_runs(self):
        """Test que quick_preprocess s'exécute sans erreur"""
        # Ce test nécessite les vraies données
        df = quick_preprocess(min_minutes=60, save=False)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Vérifier les colonnes importantes
        required_cols = [
            'web_name', 'position', 'team_name', 'price',
            'goals_per_90', 'assists_per_90', 'points_per_million'
        ]
        
        for col in required_cols:
            assert col in df.columns
    
    def test_processed_data_saved(self):
        """Test que les données preprocessées sont sauvegardées"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        filepath = os.path.join(project_root, "data", "processed", "players_cleaned.csv")
        
        assert os.path.exists(filepath), "Le fichier players_cleaned.csv n'existe pas"
        
        # Charger et vérifier
        df = pd.read_csv(filepath)
        assert len(df) > 0
        assert 'predicted_points' not in df.columns  # Pas encore de prédictions


class TestFeatureEngineering:
    """Tests spécifiques pour le feature engineering"""
    
    def test_price_conversion(self):
        """Test la conversion du prix"""
        df = pd.DataFrame({'now_cost': [130, 150, 40]})
        df['price'] = df['now_cost'] / 10
        
        assert df.iloc[0]['price'] == 13.0
        assert df.iloc[1]['price'] == 15.0
        assert df.iloc[2]['price'] == 4.0
    
    def test_points_per_million(self):
        """Test le calcul de points per million"""
        df = pd.DataFrame({
            'total_points': [100, 120, 50],
            'price': [13.0, 15.0, 5.0]
        })
        
        df['points_per_million'] = df['total_points'] / df['price']
        
        assert abs(df.iloc[0]['points_per_million'] - 7.69) < 0.01
        assert abs(df.iloc[1]['points_per_million'] - 8.0) < 0.01
        assert abs(df.iloc[2]['points_per_million'] - 10.0) < 0.01
    
    def test_no_division_by_zero(self):
        """Test qu'il n'y a pas de division par zéro"""
        df = pd.DataFrame({
            'goals_scored': [10],
            'minutes': [0]
        })
        
        df['goals_per_90'] = np.where(
            df['minutes'] > 0,
            (df['goals_scored'] * 90) / df['minutes'],
            0
        )
        
        # Ne doit pas générer d'erreur
        assert df.iloc[0]['goals_per_90'] == 0


# Fonction pour lancer les tests si ce fichier est exécuté directement
if __name__ == "__main__":
    pytest.main([__file__, "-v"])