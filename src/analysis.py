import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#réglages graphiques
sns.set(style="whitegrid", context="notebook")
plt.rcParams['figure.figsize'] = (10, 6)

def load_players(path: str= '../data/players_with_fixtures.csv'):
    """Charger les données des joueurs depuis un fichier CSV."""
    df = pd.read_csv(path)
    print(f"Données des joueurs chargées : {path} ({len(df)} lignes, {len(df.columns)} colonnes)")
    display_head = df.head().T 
    print("\nAperçu (trasposé) : \n", display_head)
    return df

def basic_checks(df: pd.DataFrame):
    """Effectuer des vérifications de base sur le DataFrame."""
    print("\nInfos générales")
    print(df.info())
    
    print("\nStatistiques descriptives :")
    print(df.describe().T)
    
    missing = df.isna().sum()
    print("\nValeurs manquantes par colonne :")
    print(missing[missing > 0])

    for col in ["prix", "points", "minutes_jouees","buts", "passes", "bonus", "influence", "creativity", "threat", "forme"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def quick_plots(df: pd.DataFrame, outdir= "../data/plots"):
    """Générer des graphiques rapides pour l'analyse exploratoire des données."""
    os.makedirs(outdir, exist_ok=True)
    
    #distribution des points
    plt.figure()
    sns.histplot(df["points"], bins=30, kde=True)
    plt.title("Distribution des points")
    plt.xlabel("Points totaux")
    plt.savefig(os.path.join(outdir, "dist_points.png"))
    plt.close()

    #points vs prix
    if "prix" in df.columns:
        plt.figure()
        sns.scatterplot(data=df, x="prix", y="points", hue="poste", alpha=0.7)
        plt.title("Points vs Prix des joueurs (par poste)")
        plt.xlabel("Prix (en millions)")
        plt.ylabel("Points")
        plt.legend(loc='best', fontsize='small')
        plt.savefig(os.path.join(outdir, "points_vs_prix.png"))
        plt.close()
        
    # Boxplot des points par poste
    if "poste" in df.columns:
        plt.figure()
        sns.boxplot(data=df, x="poste", y="points")
        plt.title("Distribution des points par poste")
        plt.xlabel("Poste")
        plt.ylabel("Points")
        plt.savefig(os.path.join(outdir, "boxplot_points_by_poste.png"))
        plt.close()

    # Corrélation entre les variables numériques
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        plt.figure(figsize=(12, 10))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title("Matrice de corrélation des variables numériques")
        plt.savefig(os.path.join(outdir, "correlation_matrix.png"))
        plt.close()

    print(f"Graphiques sauvegardés dans le dossier : {outdir}")

def feature_engineering(df: pd.DataFrame):
    """Créer des features utiles et nettoyer les colonnes pour le ML.
    - Domicile : 1 = domicile, 0 = extérieur
    - Ratio points/minutes : points / minutes jouées
    - ICT : influence + creativity + threat
    - Point par million : déja calulé, sinon calculer
    """
    df_fe = df.copy()
    
    # Domicile en binaire
    if "domicile" in df_fe.columns:
        df_fe["domicile"] = df_fe["domicile"].astype(bool).astype(int)
    else:
        df_fe["domicile"] = 0

    # Ratio points/minutes
    df_fe["minutes_jouees"] = pd.to_numeric(df_fe.get("minutes_jouees", 0), errors="coerce").fillna(0)
    df_fe["points"] = pd.to_numeric(df_fe.get("points", 0), errors="coerce").fillna(0)
    df_fe["points_per_min"] = df_fe.apply(
        lambda r: r["points"] / r["minutes_jouees"] if r["minutes_jouees"] > 0 else 0, axis=1
    )

    # ICT
    if all(c in df_fe.columns for c in ["influence", "creativity", "threat"]):
        df_fe["ict_calc"] = df_fe[["influence", "creativity", "threat"]].sum(axis=1)
    else:
        df_fe["ict_calc"] = 0

    # Point par million si pas déjà présent
    if "point_par_million" not in df_fe.columns and "prix" in df_fe.columns:
        df_fe["point_par_million"] = df_fe["point"] / df_fe["prix"].replace(0, np.nan)
        df_fe["point_par_million"].fillna(0, inplace=True)

    if "poste" in df_fe.columns:
        dummies = pd.get_dummies(df_fe["poste"], prefix="poste")
        df_fe = pd.concat([df_fe, dummies], axis=1)

    # Normalisation pour certaines colonnes (min-max) pour inspection 
    for c in ["prix", "points", "minutes_jouees", "buts", "passes", "bonus", "forme", "points_per_min", "ict_calc"]:
        if c in df_fe.columns:
            col = df_fe[c].astype(float)
            if col.max() != col.min():
                df_fe[c + "_scaled"] = (col - col.min()) / (col.max() - col.min())
            else:
                df_fe[c + "_scaled"] = 0.0
    return df_fe

def save_processed(df: pd.DataFrame, out_path: str= "../data/players_processed.csv"):
    """Sauvegarder le DataFrame traité dans un fichier CSV et prêt pour le ML."""
    df.to_csv(out_path, index=False)
    print(f"Données traitées sauvegardées dans : {out_path}")

def run_all():
    """Exécuter toutes les étapes d'analyse des données."""
    df = load_players()
    df = basic_checks(df)
    quick_plots(df)
    df_fe = feature_engineering(df)
    save_processed(df_fe)
    print("Analyse et feature engineering terminés.")
        
if __name__ == "__main__":
    run_all()