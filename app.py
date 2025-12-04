"""
FPL Points Predictor - Interface Web avec Streamlit
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path



# Page configuration
st.set_page_config(
    page_title="FPL Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def load_predictions():
    
    predictions_dir = Path("data/predictions")
    
    #direction des fichiers
    ml_path = predictions_dir / "predictions_ml.csv"
    formula_path = predictions_dir / "predictions_formula.csv"
    
    if not ml_path.exists() or not formula_path.exists():
        with st.spinner("Generating predictions... This may take 2 minutes."):
            import subprocess
            subprocess.run(["python", "main.py"], check=True)
    
    df_ml = pd.read_csv(ml_path)
    df_formula = pd.read_csv(formula_path)
    
    return df_ml, df_formula


def display_header():
    """Header"""
    st.title("FPL Points Predictor")
    st.markdown("### Machine Learning vs Weighted Formula Ranking")
    st.markdown("---")
    
    
def display_comparison(df_ml, df_formula, n=20):
    """Affiche la comparaison des prédictions"""
    st.subheader(f"Top {n} Players - Predictions Comparison")
    
    # Création de deux colonnes
    col1, col2 = st.columns(2)
    
    # Prédictions Machine Learning
    with col1:
        st.markdown("#### Machine Learning Ensemble")
        st.markdown("*XGBoost (50%) + Random Forest (30%) + Gradient Boosting (20%)*")
        
        top_ml = df_ml.nlargest(n, 'predicted_points')
        
        # Tableau résumé
        summary_ml = top_ml[['web_name', 'position', 'team_name', 'price', 'predicted_points']].copy()
        summary_ml.columns = ['Name', 'Pos', 'Team', 'Price (£M)', 'Pred Pts']
        summary_ml['Price (£M)'] = summary_ml['Price (£M)'].apply(lambda x: f"{x:.1f}")
        summary_ml['Pred Pts'] = summary_ml['Pred Pts'].apply(lambda x: f"{x:.2f}")
        summary_ml.index = range(1, len(summary_ml) + 1)
        
        st.dataframe(summary_ml, width="stretch", height=400)
        
        st.markdown("---")
        st.markdown("**Click on a player for details:**")
        
        # Display de chaque joueur cliquable
        for idx, (_, row) in enumerate(top_ml.iterrows(), 1):
            with st.expander(f"{idx} . {row['web_name']} - {row['team_name']} ({row['position']}) - £{row['price']:.1f}M - {row['total_points']:.2f} pts"):
                # Détails du joueur
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.metric("Total Points (Season)", int(row['total_points']))
                    st.metric("Form (30 days)", f"{row['form']:.1f}")
                    st.metric("Minutes Played", int(row['minutes']))
                    st.metric("Goals per 90", f"{row.get('goals_per_90', 0):.2f}")
                
                with detail_col2:
                    st.metric("Predicted Points", f"{row['predicted_points']:.2f}")
                    st.metric("Price", f"£{row['price']:.1f}M")
                    st.metric("Next Opponent", row.get('next_fixture_opponent', 'N/A'))
                    st.metric("Fixture Difficulty", int(row.get('next_fixture_difficulty', 3)))
                
    
    # Prédictions formule pondérée
    with col2:
        st.markdown("#### Weighted Formula")
        st.markdown("*35% Form + 25% xG/xA + 20% Reliability + 10% Fixture + 10% Momentum*")
        
        top_formula = df_formula.nlargest(n, 'predicted_points')
        
        # Tableau résumé
        summary_formula = top_formula[['web_name', 'position', 'team_name', 'price', 'predicted_points']].copy()
        summary_formula.columns = ['Name', 'Pos', 'Team', 'Price (£M)', 'Pred Pts']
        summary_formula['Price (£M)'] = summary_formula['Price (£M)'].apply(lambda x: f"{x:.1f}")
        summary_formula['Pred Pts'] = summary_formula['Pred Pts'].apply(lambda x: f"{x:.2f}")
        summary_formula.index = range(1, len(summary_formula) + 1)
        
        st.dataframe(summary_formula, width="stretch", height=400)
        
        st.markdown("---")
        st.markdown("**Click on a player for details:**")
        
        # Display de chaque joueur cliquable
        for idx, (_, row) in enumerate(top_formula.iterrows(), 1):
            with st.expander(f"{idx} . {row['web_name']} - {row['team_name']} ({row['position']}) - £{row['price']:.1f}M - {row['total_points']:.2f} pts"):
                # Détails du joueur
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.metric("Total Points (Season)", int(row['total_points']))
                    st.metric("Form (30 days)", f"{row['form']:.1f}")
                    st.metric("Minutes Played", int(row['minutes']))
                    st.metric("Goals per 90", f"{row.get('goals_per_90', 0):.2f}")
                
                with detail_col2:
                    st.metric("Predicted Points", f"{row['predicted_points']:.2f}")
                    st.metric("Price", f"£{row['price']:.1f}M")
                    st.metric("Next Opponent", row.get('next_fixture_opponent', 'N/A'))
                    st.metric("Fixture Difficulty", int(row.get('next_fixture_difficulty', 3)))
    
    # Analise
    st.markdown("---")
    
    # Joueurs communs dans le top 10
    ml_players = set(top_ml['web_name'].head(10))
    formula_players = set(top_formula['web_name'].head(10))
    common = ml_players & formula_players
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Common Players (Top 10)", f"{len(common)}/10")
    
    with col2:
        st.metric("Consistency", f"{len(common)*10}%")
    
    with col3:
        if len(common) >= 8:
            st.success("High Agreement")
        elif len(common) >= 6:
            st.warning("Moderate Agreement")
        elif len(common) >= 4:
            st.info("Low Agreement")
        else:
            st.error("Bad Agreement")


def display_eda():
    """Figures EDA avec descriptions"""
    st.markdown("---")
    st.subheader("Exploratory Data Analysis")
    
    figures_dir = Path("results/figures")
    
    if not figures_dir.exists():
        st.warning("Run 'python main.py' pour générer les graphiques EDA.")
        return
    
    # Liste des figures avec descriptions
    figures = [
        {
            "file": "points_distribution_by_position.png",
            "title": "Points Distribution by Position",
            "description": "Boxplot showing the distribution of total points across different positions. Midfielders and forwards typically score more points than defenders and goalkeepers."
        },
        {
            "file": "top_15_players.png",
            "title": "Top 15 Players by Total Points",
            "description": "Highest-scoring players of the current season. These players have consistently delivered points throughout the season."
        },
        {
            "file": "price_vs_points.png",
            "title": "Price vs Points Analysis",
            "description": "Scatter plot examining the relationship between player price and total points. Premium players generally deliver more points but value picks exist."
        },
        {
            "file": "correlation_matrix.png",
            "title": "Feature Correlation Matrix",
            "description": "Heatmap showing correlations between key features. Strong correlations between form, ICT index, and total points are evident."
        },
        {
            "file": "goals_assists_per_90.png",
            "title": "Goals and Assists per 90 Minutes",
            "description": "Average goals and assists per 90 minutes by position. Forwards lead in goals per 90, while midfielders contribute more assists."
        },
        {
            "file": "top_15_value_players.png",
            "title": "Top 15 Value Players",
            "description": "Best points-per-million ratio players. These players offer excellent value and are good options for budget-conscious managers."
        },
        {
            "file": "fixture_difficulty_distribution.png",
            "title": "Fixture Difficulty Distribution",
            "description": "Distribution of upcoming fixture difficulties. Lower values indicate easier matches, which often correlate with higher predicted points."
        }
    ]
    
    # Display in 2 columns
    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(figures):
                fig = figures[idx]
                fig_path = figures_dir / fig["file"]
                
                if fig_path.exists():
                    with col:
                        st.markdown(f"**{fig['title']}**")
                        st.image(str(fig_path), width="stretch")
                        st.caption(fig['description'])
                        st.markdown("")


def main():
    """Main application."""
    display_header()
    
    # Load data
    df_ml, df_formula = load_predictions()
    
    if df_ml is None or df_formula is None:
        st.stop()
    
    # Choix du filtre de position
    col1, col2 = st.columns([2, 1])
    
    with col1:
        n = st.slider("Number of players to display", min_value=10, max_value=50, value=20, step=5)
    
    with col2:
        positions = ['All'] + sorted(df_ml['position'].unique().tolist())
        selected_position = st.selectbox("Filter by position", positions)
    
    # Filtre pour position
    if selected_position != 'All':
        df_ml = df_ml[df_ml['position'] == selected_position].copy()
        df_formula = df_formula[df_formula['position'] == selected_position].copy()

    # Display pour Comparaison des prédictions
    display_comparison(df_ml, df_formula, n)
    
    # Display EDA
    display_eda()
    
    # Footer
    st.markdown("---")
    st.markdown("**FPL Points Predictor** | HEC Lausanne | Datascience and Advanced Programming 2025-2026")


if __name__ == "__main__":
    main()