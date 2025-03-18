import pandas as pd
from nba_api.stats.endpoints import leaguestandings, leaguedashteamstats
from time import sleep

start_season = 1970
end_season = 2024
# 21-22 7,9

all_playoff_teams = []

for season in range(start_season, end_season):
    season_str = f"{season}-{str(season+1)[-2:]}"
    print(f"search {season_str} data...")

    try:
        standings = leaguestandings.LeagueStandings(season=season_str).get_data_frames()[0]

        playoff_teams = standings[standings['PlayoffRank'] > 0][['TeamID', 'TeamName', 'Conference', 'PlayoffRank']]
        playoff_teams['SEASON'] = season_str

        stats = leaguedashteamstats.LeagueDashTeamStats(season=season_str).get_data_frames()[0]
        stats = stats[['TEAM_ID', 'W_PCT', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST', 'REB', 'TOV', 'STL', 'BLK']]

        playoff_teams = playoff_teams.merge(stats, left_on='TeamID', right_on='TEAM_ID', how='left', suffixes=('', '_stats'))
        if(season_str == '2021-22'):
            playoff_teams = playoff_teams[playoff_teams['PlayoffRank'] < 10]
        else:
            playoff_teams = playoff_teams[playoff_teams['PlayoffRank'] < 9]
        playoff_teams.drop(columns=['TEAM_ID'], inplace=True)
        playoff_teams.drop(columns=['TeamID'], inplace=True)
        all_playoff_teams.append(playoff_teams)

        sleep(1)

    except Exception as e:
        print(f"Wrong: {season_str}. error message: {e}")

df_playoff_teams = pd.concat(all_playoff_teams, ignore_index=True)

df_playoff_teams.to_csv("../data/data.csv", index=False)
print("saved as nba_playoff_teams.csv")
