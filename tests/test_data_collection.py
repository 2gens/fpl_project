"""
Tests avec pytest pour vérifier que la collecte fonctionne correctement.
"""

import pytest
import pandas as pd
import json
import os
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection import FPLDataCollector, quick_collect, load_latest_data


class TestFPLDataCollector:
    """Tests pour la classe FPLDataCollector"""
    
    def test_collector_initialization(self):
        """Test que le collector s'initialise correctement"""
        collector = FPLDataCollector()
        assert collector is not None
        assert collector.base_url == "https://fantasy.premierleague.com/api"
        assert collector.session is not None
    
    def test_fetch_bootstrap_static(self):
        """Test que fetch_bootstrap_static retourne des données valides"""
        collector = FPLDataCollector()
        data = collector.fetch_bootstrap_static()
        
        assert data is not None
        assert 'elements' in data
        assert 'teams' in data
        assert 'events' in data
        assert len(data['elements']) > 0
        assert len(data['teams']) == 20  # 20 équipes en Premier League
    
    def test_fetch_fixtures(self):
        """Test que fetch_fixtures retourne des fixtures valides"""
        collector = FPLDataCollector()
        fixtures = collector.fetch_fixtures()
        
        assert fixtures is not None
        assert isinstance(fixtures, list)
        assert len(fixtures) > 0
        
        # Vérifier la structure d'un fixture
        fixture = fixtures[0]
        assert 'team_h' in fixture
        assert 'team_a' in fixture
        assert 'event' in fixture
    
    def test_load_players_dataframe(self):
        """Test la conversion des données en DataFrame"""
        collector = FPLDataCollector()
        bootstrap_data = collector.fetch_bootstrap_static()
        
        df = collector.load_players_dataframe(bootstrap_data)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'web_name' in df.columns
        assert 'total_points' in df.columns
        assert 'now_cost' in df.columns
    
    def test_get_teams_dict(self):
        """Test la création du dictionnaire d'équipes"""
        collector = FPLDataCollector()
        bootstrap_data = collector.fetch_bootstrap_static()
        
        teams_dict = collector.get_teams_dict(bootstrap_data)
        
        assert teams_dict is not None
        assert isinstance(teams_dict, dict)
        assert len(teams_dict) == 20
        
        # Vérifier que les valeurs sont des strings 
        for team_id, team_name in teams_dict.items():
            assert isinstance(team_id, int)
            assert isinstance(team_name, str)
    
    def test_get_positions_dict(self):
        """Test la création du dictionnaire de positions"""
        collector = FPLDataCollector()
        bootstrap_data = collector.fetch_bootstrap_static()
        
        positions_dict = collector.get_positions_dict(bootstrap_data)
        
        assert positions_dict is not None
        assert isinstance(positions_dict, dict)
        assert len(positions_dict) == 4  # GK, DEF, MID, FWD
        
        # Vérifier les positions attendues
        expected_positions = ['GKP', 'DEF', 'MID', 'FWD']
        actual_positions = list(positions_dict.values())
        
        for pos in expected_positions:
            assert pos in actual_positions
    
    def test_get_next_gameweek(self):
        """Test la détection de la prochaine gameweek"""
        collector = FPLDataCollector()
        bootstrap_data = collector.fetch_bootstrap_static()
        
        gameweek_info = collector.get_next_gameweek(bootstrap_data)
        
        assert gameweek_info is not None
        assert 'current_gw' in gameweek_info
        assert 'next_gw' in gameweek_info
        assert 'status' in gameweek_info
        assert 'next_deadline' in gameweek_info
        
        # Vérifier que les numéros de gameweek sont valides
        assert 1 <= gameweek_info['current_gw'] <= 38
        assert 1 <= gameweek_info['next_gw'] <= 38


class TestDataCollection:
    """Tests pour les fonctions utilitaires"""
    
    def test_quick_collect(self):
        """Test la fonction quick_collect"""
        data = quick_collect()
        
        assert data is not None
        assert 'bootstrap' in data
        assert 'fixtures' in data
        assert 'gameweek_info' in data
        assert 'collected_at' in data
    
    def test_data_saved_after_collection(self):
        """Test que les données sont bien sauvegardées"""
        # Vérifier que le fichier existe
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        filepath = os.path.join(project_root, "data", "raw", "fpl_latest.json")
        
        assert os.path.exists(filepath), "Le fichier fpl_latest.json n'existe pas"
        
        # Charger et vérifier le contenu
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'bootstrap' in data
        assert 'fixtures' in data


# Fonction pour lancer les tests si ce fichier est exécuté directement
if __name__ == "__main__":
    pytest.main([__file__, "-v"])