"""
Ce module crée des visualisations pour comprendre les données FPL.
Graphiques générés :
- Distribution des points par position
- Top joueurs par points et par value
- Corrélations entre variables
- Prix vs Points
- Goals/Assists per 90 par position
- Fixture difficulty analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, List
from datetime import datetime


class FPLExploratoryAnalysis:
  
    def __init__(self, output_dir: str = None):

        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            output_dir = os.path.join(project_root, "results", "figures")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Style des graphiques
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        print(f"EDA initialisé - Figures sauvegardées dans : {output_dir}")
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Charger les données preprocessées.
        """
        if filepath is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            filepath = os.path.join(project_root, "data", "processed", "players_cleaned.csv")
        
        print(f"Chargement des données depuis : {filepath}")
        df = pd.read_csv(filepath)
        print(f"Données chargées : {len(df)} joueurs")
        
        return df
    
    def plot_points_distribution_by_position(self, df: pd.DataFrame, save: bool = True):
        """
        Graphique : Distribution des points par position.
        """
        print("\nCréation du graphique : Distribution des points par position")
        
        fig, ax = plt.subplots(figsize=(12, 6))


        # Couleur de fond personnalisée
        bg_color1 = "#dfe9e4ff"          
        fig.patch.set_facecolor(bg_color1)  
        ax.set_facecolor("#dfe9e4ff")
        
        # Boxplot par position
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        data_by_position = [df[df['position'] == pos]['total_points'] for pos in positions]
        
        bp = ax.boxplot(data_by_position, labels=positions, patch_artist=True)
        
        # Colorer les boîtes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Distribution des Points Totaux par Position', fontsize=16, fontweight='bold')
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Points Totaux', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'points_distribution_by_position.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=bg_color1)
            print(f"Figure sauvegardée : {filepath}")
        
        plt.close()
    
    def plot_top_players(self, df: pd.DataFrame, n: int = 15, save: bool = True):
        """
        Graphique : Top N joueurs par points totaux.
        """
        print(f"\nCréation du graphique : Top {n} joueurs par points")
        
        top_players = df.nlargest(n, 'total_points')[['web_name', 'total_points', 'position']].copy()
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # Couleur de fond personnalisée
        bg_color2 = "#dfe9e4ff"          
        fig.patch.set_facecolor(bg_color2)  
        ax.set_facecolor("#dfe9e4ff")
        
        # Créer un barplot horizontal
        colors_map = {'GKP': '#FF6B6B', 'DEF': '#4ECDC4', 'MID': '#45B7D1', 'FWD': '#FFA07A'}
        colors = [colors_map.get(pos, '#95A5A6') for pos in top_players['position']]
        
        bars = ax.barh(range(len(top_players)), top_players['total_points'], color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_players)))
        ax.set_yticklabels(top_players['web_name'])
        ax.set_xlabel('Points Totaux', fontsize=12)
        ax.set_title(f'Top {n} Joueurs par Points Totaux', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, (bar, points) in enumerate(zip(bars, top_players['total_points'])):
            ax.text(points + 1, i, f'{points}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'top_{n}_players.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=bg_color2)
            print(f"Figure sauvegardée : {filepath}")
        
        plt.close()
    
    def plot_price_vs_points(self, df: pd.DataFrame, save: bool = True):
        """
        Graphique : Prix vs Points (scatter plot).
        """
        print("\nCréation du graphique : Prix vs Points")
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # Couleur de fond personnalisée
        bg_color3 = "#dfe9e4ff"          
        fig.patch.set_facecolor(bg_color3)  
        ax.set_facecolor("#dfe9e4ff")
        
        # Scatter plot par position
        positions = df['position'].unique()
        colors_map = {'GKP': '#FF6B6B', 'DEF': '#4ECDC4', 'MID': '#45B7D1', 'FWD': '#FFA07A'}
        
        for position in positions:
            pos_data = df[df['position'] == position]
            ax.scatter(pos_data['price'], pos_data['total_points'], 
                      label=position, alpha=0.6, s=50, 
                      color=colors_map.get(position, '#95A5A6'))
        
        ax.set_xlabel('Prix (£M)', fontsize=12)
        ax.set_ylabel('Points Totaux', fontsize=12)
        ax.set_title('Relation Prix vs Points par Position', fontsize=16, fontweight='bold')
        ax.legend(title='Position', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'price_vs_points.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=bg_color3)
            print(f"Figure sauvegardée : {filepath}")
        
        plt.close()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True):
        """
        Graphique : Matrice de corrélation des variables importantes.
        """
        print("\nCréation du graphique : Matrice de corrélation")
        
        numeric_cols = [
            'total_points', 'price', 'form', 'minutes',
            'goals_per_90', 'assists_per_90', 'ict_index',
            'next_fixture_difficulty', 'avg_fixture_difficulty_5'
        ]
        
        # Garder seulement les colonnes qui existent
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        corr_matrix = df[available_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))

        # Couleur de fond personnalisée
        bg_color4 = "#dfe9e4ff"          
        fig.patch.set_facecolor(bg_color4)  
        ax.set_facecolor("#dfe9e4ff")
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Matrice de Corrélation des Variables', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'correlation_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=bg_color4)
            print(f"Figure sauvegardée : {filepath}")
        
        plt.close()
    
    def plot_goals_assists_per_90(self, df: pd.DataFrame, save: bool = True):
        """
        Graphique : Goals et Assists per 90 par position.
        """
        print("\nCréation du graphique : Goals et Assists per 90 par position")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Couleur de fond personnalisée
        bg_color5 = "#dfe9e4ff"          
        fig.patch.set_facecolor(bg_color5)  
        ax1.set_facecolor("#dfe9e4ff")
        ax2.set_facecolor("#dfe9e4ff")
        
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        
        # Goals per 90
        goals_by_pos = [df[df['position'] == pos]['goals_per_90'].mean() for pos in positions]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        bars1 = ax1.bar(positions, goals_by_pos, color=colors, alpha=0.8)
        ax1.set_title('Goals Moyens per 90 par Position', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Goals per 90', fontsize=12)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Ajouter les valeurs
        for bar, val in zip(bars1, goals_by_pos):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Assists per 90
        assists_by_pos = [df[df['position'] == pos]['assists_per_90'].mean() for pos in positions]
        
        bars2 = ax2.bar(positions, assists_by_pos, color=colors, alpha=0.8)
        ax2.set_title('Assists Moyens per 90 par Position', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Assists per 90', fontsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Ajouter les valeurs
        for bar, val in zip(bars2, assists_by_pos):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'goals_assists_per_90.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=bg_color5)
            print(f"Figure sauvegardée : {filepath}")
        
        plt.close()
    
    def plot_value_analysis(self, df: pd.DataFrame, n: int = 15, save: bool = True):
        """
        Graphique : Top joueurs par "value" (points per million).
        """
        print(f"\nCréation du graphique : Top {n} joueurs par value")
        
        top_value = df.nlargest(n, 'points_per_million')[
            ['web_name', 'points_per_million', 'position', 'price']
        ].copy()
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # Couleur de fond personnalisée
        bg_color6 = "#dfe9e4ff"          
        fig.patch.set_facecolor(bg_color6)  
        ax.set_facecolor("#dfe9e4ff")
        
        colors_map = {'GKP': '#FF6B6B', 'DEF': '#4ECDC4', 'MID': '#45B7D1', 'FWD': '#FFA07A'}
        colors = [colors_map.get(pos, '#95A5A6') for pos in top_value['position']]
        
        bars = ax.barh(range(len(top_value)), top_value['points_per_million'], 
                      color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_value)))
        ax.set_yticklabels([f"{name} (£{price:.1f}M)" 
                           for name, price in zip(top_value['web_name'], top_value['price'])])
        ax.set_xlabel('Points per Million', fontsize=12)
        ax.set_title(f'Top {n} Meilleurs "Value" Players', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Ajouter les valeurs
        for i, (bar, val) in enumerate(zip(bars, top_value['points_per_million'])):
            ax.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'top_{n}_value_players.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=bg_color6)
            print(f"Figure sauvegardée : {filepath}")
        
        plt.close()
    
    def plot_fixture_difficulty(self, df: pd.DataFrame, save: bool = True):
        """
        Graphique : Distribution de la difficulté des fixtures.
        """
        print("\nCréation du graphique : Distribution fixture difficulty")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Couleur de fond personnalisée
        bg_color7 = "#dfe9e4ff"          
        fig.patch.set_facecolor(bg_color7)  
        ax1.set_facecolor("#dfe9e4ff")  
        ax2.set_facecolor("#dfe9e4ff") 
        
        # Next fixture difficulty
        if 'next_fixture_difficulty' in df.columns:
            difficulty_counts = df['next_fixture_difficulty'].value_counts().sort_index()
            
            colors = ["#2DC16A", '#F39C12', '#E67E22', '#E74C3C', '#C0392B']
            bars1 = ax1.bar(difficulty_counts.index, difficulty_counts.values, 
                           color=colors, alpha=0.8)
            
            ax1.set_xlabel('Difficulté (1=Facile, 5=Difficile)', fontsize=12)
            ax1.set_ylabel('Nombre de Joueurs', fontsize=12)
            ax1.set_title('Distribution de la Difficulté du Prochain Match', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Ajouter les valeurs
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Average fixture difficulty over 5 games
        if 'avg_fixture_difficulty_5' in df.columns:
            ax2.hist(df['avg_fixture_difficulty_5'], bins=20, color='#3498DB', 
                    alpha=0.7, edgecolor='black')
            
            ax2.set_xlabel('Difficulté Moyenne (5 prochains matchs)', fontsize=12)
            ax2.set_ylabel('Nombre de Joueurs', fontsize=12)
            ax2.set_title('Distribution de la Difficulté Moyenne sur 5 Matchs', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, axis='y', alpha=0.3)
            
            # Ajouter une ligne verticale pour la moyenne
            mean_diff = df['avg_fixture_difficulty_5'].mean()
            ax2.axvline(mean_diff, color='red', linestyle='--', linewidth=2, 
                       label=f'Moyenne: {mean_diff:.2f}')
            ax2.legend()
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'fixture_difficulty_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=bg_color7)
            print(f"Figure sauvegardée : {filepath}")
        
        plt.close()
    
    def generate_all_plots(self, df: pd.DataFrame):
        """
        Génère tous les graphiques.
        """
        
        print("Génération de tous les graphiques EDA.")
        
        self.plot_points_distribution_by_position(df)
        self.plot_top_players(df, n=15)
        self.plot_price_vs_points(df)
        self.plot_correlation_matrix(df)
        self.plot_goals_assists_per_90(df)
        self.plot_value_analysis(df, n=15)
        self.plot_fixture_difficulty(df)
        
        print(f"Figures sauvegardées dans : {self.output_dir}")
        


# Fonction utilitaire

def quick_eda():
    
    print("Analyse exploratoire des données FPL")
    
    eda = FPLExploratoryAnalysis()
    
    # Charger les données
    df = eda.load_data()
    
    # Générer tous les graphiques
    eda.generate_all_plots(df)
    
    return eda