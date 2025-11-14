import requests
import pandas as pd

# 1. Récupérer les fixtures
url_fixtures = "https://fantasy.premierleague.com/api/fixtures/"
fixtures = requests.get(url_fixtures).json()
df_fixtures = pd.DataFrame(fixtures)
df_fixtures.to_csv("../data/fixtures_cleaned.csv", index=False)

# 2. Filtrer les colonnes nécessaires
cols_needed = ['event','team_h','team_a','kickoff_time','team_h_difficulty','team_a_difficulty']
df_fixtures = df_fixtures[[c for c in cols_needed if c in df_fixtures.columns]]

# 3. Récupérer les équipes
url_bootstrap = "https://fantasy.premierleague.com/api/bootstrap-static/"
bootstrap = requests.get(url_bootstrap).json()
df_teams = pd.DataFrame(bootstrap['teams'])[['id','name']]

# 4. Merge pour team_h
df_fixtures = df_fixtures.merge(df_teams, left_on='team_h', right_on='id', how='left', suffixes=('', '_team'))
df_fixtures.rename(columns={'name':'team_home'}, inplace=True)
df_fixtures.drop(columns=[col for col in ['team_h','id'] if col in df_fixtures.columns], inplace=True)

# 5. Merge pour team_a
df_fixtures = df_fixtures.merge(df_teams, left_on='team_a', right_on='id', how='left', suffixes=('', '_team'))
df_fixtures.rename(columns={'name':'team_away'}, inplace=True)
df_fixtures.drop(columns=[col for col in ['team_a','id'] if col in df_fixtures.columns], inplace=True)

# 6. Tri et sauvegarde
df_fixtures.sort_values(by='event', inplace=True)
df_fixtures.to_csv("fixtures_cleaned.csv", index=False)
print(f"Fixtures sauvegardées : {len(df_fixtures)} lignes. Colonnes : {list(df_fixtures.columns)}")