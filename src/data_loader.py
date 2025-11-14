import pandas as pd
from datetime import datetime

def load_data(players_path: str, fixtures_path: str):
    """
    Charge les fichiers nettoyés (joueurs + fixtures) depuis le dossier data.
    """
    try:
        players = pd.read_csv(players_path)
        fixtures = pd.read_csv(fixtures_path)
        print(f" Données chargées : {len(players)} joueurs, {len(fixtures)} matchs à venir")
        return players, fixtures
    except FileNotFoundError:
        print("Erreur : impossible de trouver les fichiers CSV. Vérifie ton dossier /data/")
        return None, None


def merge_players_with_fixtures(players_df: pd.DataFrame, fixtures_df: pd.DataFrame):
    """
    Fusionne les joueurs avec leurs prochains adversaires.
    Ajoute :
      - prochain_adversaire : nom de l’équipe adverse
      - domicile : True si le joueur joue à domicile
      - fixture_difficulty : difficulté du prochain match pour le joueur selon l’adversaire
    """
    merged_data = players_df.copy()
    merged_data['prochain_adversaire'] = None
    merged_data['domicile'] = None
    merged_data['fixture_difficulty'] = -1  # valeur par défaut

    # --- Conversion de kickoff_time en datetime et suppression du fuseau horaire ---
    fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time']).dt.tz_convert(None)
    now = pd.Timestamp.now()

    # Garder uniquement les matchs à venir
    upcoming_fixtures = fixtures_df[fixtures_df['kickoff_time'] > now]
    if upcoming_fixtures.empty:
        print("Aucun match à venir trouvé ! Vérifie fixtures_cleaned.csv")
        return merged_data

    # On prend la première Game Week à venir
    next_gw = upcoming_fixtures['event'].min()
    next_fixtures = upcoming_fixtures[upcoming_fixtures['event'] == next_gw]

    # Boucle sur chaque match de la prochaine Game Week
    for _, match in next_fixtures.iterrows():
        home_team = match['team_home']
        away_team = match['team_away']
        home_diff = match.get('team_h_difficulty', None)
        away_diff = match.get('team_a_difficulty', None)

        # Pour les joueurs de l’équipe à domicile
        mask_home = merged_data['equipe'] == home_team
        merged_data.loc[mask_home, 'prochain_adversaire'] = away_team
        merged_data.loc[mask_home, 'domicile'] = True
        if pd.notna(away_diff):
            # Difficulté pour le joueur = inversé par rapport à la difficulté de l’adversaire
            merged_data.loc[mask_home, 'fixture_difficulty'] = 6 - away_diff

        # Pour les joueurs de l’équipe à l’extérieur
        mask_away = merged_data['equipe'] == away_team
        merged_data.loc[mask_away, 'prochain_adversaire'] = home_team
        merged_data.loc[mask_away, 'domicile'] = False
        if pd.notna(home_diff):
            merged_data.loc[mask_away, 'fixture_difficulty'] = 6 - home_diff

    # Nettoyage final
    merged_data['domicile'].fillna(False, inplace=True)
    merged_data['prochain_adversaire'].fillna('N/A', inplace=True)
    merged_data['fixture_difficulty'] = pd.to_numeric(
        merged_data['fixture_difficulty'], errors='coerce'
    ).fillna(-1)

    print(f"Fusion terminée : {merged_data['prochain_adversaire'].nunique()} adversaires ajoutés")
    return merged_data


def save_merged_data(merged_df: pd.DataFrame, output_path: str):
    """
    Sauvegarde le dataset fusionné.
    """
    merged_df.to_csv(output_path, index=False)
    print(f"Données fusionnées sauvegardées → {output_path}")


# Exemple d'utilisation (si tu exécutes ce fichier directement)
if __name__ == "__main__":
    players, fixtures = load_data("../data/players_cleaned.csv", "../data/fixtures_cleaned.csv")
    if players is not None and fixtures is not None:
        merged = merge_players_with_fixtures(players, fixtures)
        save_merged_data(merged, "../data/players_with_fixtures.csv")




    