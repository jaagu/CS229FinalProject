#https://github.com/seemethere/nba_py
from nba_py import team
from nba_py import game
from nba_py.constants import TEAMS
from datetime import datetime
import pandas as pd
import numpy as np
teamToIndex = {
    'ATL': 0,
    'BOS': 1,
    'BKN': 2,  
    'CHA': 3,
    'CHI': 4,
    'CLE': 5,
    'DAL': 6,
    'DEN': 7,
    'DET': 8,
    'GSW': 9,
    'HOU': 10,
    'IND': 11,
    'LAC': 12,
    'LAL': 13,
    'MEM': 14,
    'MIA': 15,
    'MIL': 16,
    'MIN': 17,
    'NOP': 18,
    'NYK': 19,
    'OKC': 20,
    'ORL': 21,
    'PHI': 22,
    'PHX': 23,
    'POR': 24,
    'SAC': 25,
    'SAS': 26,
    'TOR': 27,
    'UTA': 28,
    'WAS': 29,
}

def load_gameDataWithPlayersAndPointsBetweenYears(teamName, startYear, endYear):
    df = get_gameDataWithPlayersAndPoints(teamName, get_season(startYear))
    for i in range(endYear-startYear):
        season = get_season(startYear+1+i)
        df = df.append(get_gameDataWithPlayersAndPoints(teamName, season), ignore_index=True)
    return df

def get_gameDataWithPlayersAndPoints(teamName, season):
    team_id = TEAMS[teamName]['id']
    df = team.TeamGameLogs(team_id, season).info()
    print(df[:5])
    #   For each game get the players who played and the points and add them to the dataframe line
    for g in df["Game_ID"]:
        print(g)
        player_stats = game.BoxscoreAdvanced(g).sql_players_advanced()
        print(player_stats)

        players = player_stats["PLAYER_NAME"].values
        points = player_stats["PTS"].values
        teamStartIndices = np.where(player_stats["START_POSITION"].values == 'F')[0]
        home_team_players = players[:teamStartIndices[2]]
        away_team_players = players[:teamStartIndices[2]]

        #   Remove players who didn't play
        notPlayingPlayers = np.where(np.isnan(points))[0]
        players = np.delete(players, notPlayingPlayers)
        points = np.delete(points, notPlayingPlayers)
        print("------")
        
        break
    return df

def load_gameDataWithTeamsAndPointsBetweenYears(teamName, startYear, endYear):
    df = get_gameDataWithTeamsAndPoints(teamName, get_season(startYear))
    for i in range(endYear-startYear):
        season = get_season(startYear+1+i)
        print("---------------")
        print(season)
        df = df.append(get_gameDataWithTeamsAndPoints(teamName, season), ignore_index=True)
        print("-----------------")
    return df

def get_gameDataWithTeamsAndPoints(teamName, season):
    teamsPlayingArray = []
    teamPointsArray = []

    team_id = TEAMS[teamName]['id']
    df = team.TeamGameLogs(team_id, season).info()

    gameIds = df["Game_ID"].values
    #   For each game get the teams who played and the points and add them to the dataframe line
    i = 0
    percentDone = 0
    for matchup in df["MATCHUP"]:
        print( "%s %f " % (teamName, i/len(gameIds)))
        teamsVector = np.zeros(2*len(teamToIndex))
        pointsVector = np.zeros(2*len(teamToIndex))
        team_stats = game.Boxscore(gameIds[i]).team_stats()
        home = 0
        if '@' in matchup:
            home = 0
        else:
            home = 1
        teams = matchup.split(" ")
##        if(teams[0] not in teamToIndex.keys() or )
        t1Index = teamToIndex[teams[0]]
        t2Index = teamToIndex[teams[2]]
        if(team_stats["TEAM_ID"].values[0] == teamName):
            teamNamePoints = team_stats["PTS"].values[0]
            otherPoints = team_stats["PTS"].values[1]
        else:
            teamNamePoints = team_stats["PTS"].values[1]
            otherPoints = team_stats["PTS"].values[0]
        if home:
            teamsVector[t1Index] = 1
            teamsVector[t2Index+len(teamToIndex)] = 1
            
            pointsVector[t1Index] = teamNamePoints
            pointsVector[t2Index+len(teamToIndex)] = otherPoints
        else:
            teamsVector[t2Index] = 1
            teamsVector[t1Index+len(teamToIndex)] = 1
            pointsVector[t2Index] = otherPoints
            pointsVector[t1Index+len(teamToIndex)] = teamNamePoints
        

        teamsPlayingArray.append(teamsVector)
        teamPointsArray.append(pointsVector)
        i+=1
    df = df.assign(teams = pd.Series(teamsPlayingArray).values)
    df = df.assign(points = pd.Series(teamPointsArray).values)
    return df

#   Loads boxScores
def get_teamBoxScore(teamName, season):
    #Use nba_py to load data
    team_id = TEAMS[teamName]['id']
    df = team.TeamGameLogs(team_id, season).info()
    return df

def get_season(year):
    CURRENT_SEASON = str(year) + "-" + str(year + 1)[2:]
    return CURRENT_SEASON

#   Loads and adds games for teamName between startYear and endYear seasons to one table
def load_teamBoxScoresBetweenYears(teamName, startYear, endYear):
    df = get_teamBoxScore(teamName, get_season(startYear))
    for i in range(endYear-startYear):
        season = get_season(startYear+1+i)
        df = df.append(get_teamBoxScore(teamName, season), ignore_index=True)
    return df

