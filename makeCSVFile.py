import getData
import pandas as pd
import numpy as np


def main():
    print('ATL')
    df = getData.load_gameDataWithTeamsAndPointsBetweenYears('ATL',2013,2017)
    for team in getData.teamToIndex.keys():
        print(team)
        #load data from the internet
        if(team != 'ATL'):
            df = df.append(getData.load_gameDataWithTeamsAndPointsBetweenYears(team,2013,2017), ignore_index = True)
    df.to_csv('data.csv')
    return

if __name__ == '__main__':
    main()
