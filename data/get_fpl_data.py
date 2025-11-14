import requests 
import pandas as pd

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(url)
data = response.json()

players = data['elements']
df_players = pd.DataFrame(players)

teams = data['teams']
df_teams = pd.DataFrame(teams)[['id', 'name']]

columns_to_keep = ['id', 'first_name', 'second_name', 'team', 'element_type', 'now_cost', 'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'yellow_cards', 'red_cards', 'bonus', 'influence', 'creativity', 'threat','form']   
df_players = df_players[columns_to_keep]

existing_columns = [col for col in columns_to_keep if col in df_players.columns]
df_players = df_players[existing_columns]

df_players = df_players.merge(df_teams, left_on='team', right_on='id', how='left')
df_players.rename(columns={'name': 'equipe'}, inplace=True)
df_players.drop(columns=['team'], inplace=True)

df_players.rename(columns={
    'first_name': 'prenom',
    'second_name': 'nom',
    'team': 'equipe',
    'element_type': 'poste',
    'now_cost': 'prix',
    'total_points': 'points',
    'minutes': 'minutes_jouees',
    'goals_scored': 'buts',
    'assists': 'passes',
    'clean_sheets': 'clean_sheets',
    'yellow_cards': 'jaunes',   
    'red_cards': 'rouges',
    'form': 'forme'
}, inplace=True)

df_players.fillna(0, inplace=True)
df_players.drop_duplicates(inplace=True)
df_players['prix'] = df_players['prix'] / 10.0
df_players['points'] = df_players['points'].astype(int)

df_players['point_par_million'] = df_players['points'] / df_players['prix'].replace(0, 0.1)

df_players = df_players[df_players['points'] > 0]

df_players.sort_values(by='points', ascending=False, inplace=True)

df_players.to_csv("players_cleaned.csv", index=False)
print(f"Données FPL récupérées et sauvegardées ({len(df_players)} joueurs)")