# Data Documentation

## Data Description

| Feature Name              | Type    | Description |
|---------------------------|---------|-------------|
| TeamName                 | string  | Name of each playoff team |
| Conference               | string  | East / West |
| PlayoffRank              | int     | Rank in regular season (1 ~ 8) |
| SEASON                   | int     | Years from 1997 to 2024 |
| W_PCT                    | int     | Win % |
| PLUS_MINUS               | int     | Overall point differential in season |
| FG_PCT                   | int     | Field Goal % |
| FG3_PCT                  | int     | 3-Point Field Goal % |
| FT_PCT                   | int     | Free Throw % |
| AST                      | int     | Assist |
| REB                      | int     | Rebound |
| TOV                      | int     | Turnover |
| STL                      | int     | Steal |
| BLK                      | int     | Block |
| AllStar_count            | int     | The number of players selected as All-Stars that season (excluding those who missed the All-Star game due to injury). |
| Finals_Appearances_Last5 | int     | The number of times a team has reached the Finals in the past five years |
| ChampionNumber           | int     | The total number of championships a team has won in its history (including under previous team names). |
| Label                    | int     | Whether the team won the championship (label ratio of 0 : 1 is 420 : 28) |

## Source
- **nba_api**: A Python package that allows you to access NBA statistics and data from NBA.com.
- **Basketball Reference**: A comprehensive basketball statistics website that provides detailed historical and current NBA data.

## Dataset Size
- **Column size**: 18 (details in the table above)
- **Row size**: 448 (16 teams each year from 1997 to 2024)

## Process of Data Collection
(Execute `get_data/getTeamData.py` to complete steps 1~3)

1. Used **nba_api** to retrieve information (**Conference, TeamName, PlayoffRank, SEASON**) of the teams ranked **1st to 8th** in the regular season from **1997 to 2024**.  
   - *(Due to the play-in tournament, the 9th-seeded **Pelicans** and **Hawks** in the 2022 playoffs replaced the original 8th-seeded **Clippers** and **Cavaliers** in the Western and Eastern Conferences.)*

2. Collected team statistics using **nba_api**, including:
   - `W_PCT`, `PLUS_MINUS`, `FG_PCT`, `FG3_PCT`, `FT_PCT`, `AST`, `REB`, `TOV`, `STL`, `BLK`

3. Merged the datasets using `team_id`.
4. Modified `SEASON` label format (e.g., from **2023-24** to **2024**).
5. Used **Basketball Reference** to collect or calculate additional data for each team:
   - `AllStar_count`, `ChampionNumber`, `Finals_Appearances_Last5`, `Label`.
## Reference
- [Basketball Reference](https://www.basketball-reference.com)
- [NBA API GitHub Repository](https://github.com/swar/nba_api/tree/master/src/nba_api)

