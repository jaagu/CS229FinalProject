import getData
from sklearn import svm
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


ofeatures = ['homeAway', 'L', 'W']
doLinearRegression = False
doSVM = True

def no_shuffle_train_test_split(X,Y, percentage):
    indexFromPercentage = int(X.shape[0]*percentage)+1
    train_X, test_X = np.split(X, [indexFromPercentage])
    train_Y, test_Y = np.split(Y, [indexFromPercentage])
    return train_X, test_X, train_Y, test_Y
    

def load_data(filename, removeZeros = False):
    df = pd.read_csv(filename, header=0)
    
    if(removeZeros):
        homeScore = []
        awayScore = []
        newdf = pd.DataFrame()
        for row in df.values:
            scores = [i for i in row if i > 0]
            homeScore.append(scores[0])
            awayScore.append(scores[1])
        hs = pd.Series(homeScore)
        aways = pd.Series(awayScore)
        newdf = newdf.assign(homescores = hs.values)
        newdf = newdf.assign(awayscores = aways.values)
        return newdf.values
    return df.values

def add_homeaway_teams(df):
    home = []
    tpArray = []
    dateweightArray = []
    dates=df['GAME_DATE'].values
    i = 0
    for matchup in df['MATCHUP']:
        teamsPlaying = np.zeros(len(getData.teamToIndex))
        if '@' in matchup:
            home.append(0)
        else:
            home.append(1)
        teams = matchup.split(" ")
        t1Index = getData.teamToIndex[teams[0]]
        t2Index = getData.teamToIndex[teams[2]]
        teamsPlaying[t1Index] = 1
        teamsPlaying[t2Index] = 1
        tpArray.append(teamsPlaying)

        #add date weight
        date = dates[i].split(" ")
        dateweightArray.append(0.9**(2017-int(date[2])+1))
        i += 1
        
    ha = pd.Series(home)
    tp = pd.Series(tpArray)
    dw = pd.Series(dateweightArray)
    df = df.assign(homeAway = ha.values)
    df = df.assign(playingTeams = tp.values)
    df = df.assign(dateWeight = dw.values)
        
    return df

def winLossToBool(truth):
    for i in range(len(truth)):
        if truth[i] == 'W':
            truth[i] = 1
        else:
            truth[i] = 0
    return truth

def addOtherFeaturesToPlayingTeams(dataframe, feature, additionalfeatures):
    features = dataframe[feature].values.tolist()
    otherfeatures = dataframe[additionalfeatures].values
    for i in range(len(features)):
        featurelist = features[i].tolist()
        for j in range(len(otherfeatures[i])):
            if(otherfeatures[i][j] != 'nan'):
                featurelist.append(otherfeatures[i][j])
            else:
                print('hi')
                featurelist.append(0)
        features[i] = featurelist
    return features

def trainSVM(train):
    clf = svm.SVC(kernel='rbf')
    features = addOtherFeaturesToPlayingTeams(train,'playingTeams',ofeatures)
##    print(features)
    truth = winLossToBool(train['WL'].values.tolist())
    clf.fit(features,truth)
    return clf

def testSVM(clf, test):
    truth = winLossToBool(test['WL'].values.tolist())
    features = addOtherFeaturesToPlayingTeams(test,'playingTeams',ofeatures)
##    print(features)
    correct = 0
    total = len(test)
    for t in range(len(test)):
        if(truth[t] == 'nan'):
            print('nan')
        if(truth[t] == clf.predict([features[t]])):
           correct+=1
    return(correct/float(total))

def dealWithSVM():
    X = []
    Y = []
    trainError = []
    i = 1
    teams = []
    for team in getData.teamToIndex.keys():
        X.append(i)
        teams.append(team)
        i += 1
        print(team)
        #load data from the internet
        df = getData.load_teamBoxScoresBetweenYears(team,2013,2017)
        totalAccuracy = 0
        totalTrainAccuracy = 0
        for _ in range(100):
            #get train, test
            train, test = train_test_split(df, test_size=0.2)
            #modify data
            train = add_homeaway_teams(train)
            test = add_homeaway_teams(test)
            #train svm
            clf = trainSVM(train)
            #test svm
            totalAccuracy += testSVM(clf, test)
            totalTrainAccuracy += testSVM(clf,train)
        averageTrainError = totalTrainAccuracy/float(100)
        print(train.shape)
        averageError = totalAccuracy/float(100)
        print(averageError)
        print(averageTrainError)
        trainError.append(averageTrainError)
        Y.append(averageError)
    print("TEST ERROR: %f" % (np.mean(Y)))

    print("TRAIN ERROR: %f" % (np.mean(trainError)))
    plt.plot(X,Y, marker = 'o')
    plt.xticks(X, teams, rotation=45, fontsize = 10)
    plt.ylim(ymin=0.3, ymax=1)

    plt.plot([0, 31], [0.5,0.5], color = 'red')
    plt.grid(True)
    plt.show()
    return

def testResults(lr, X_test, Y_test):
    correct = 0
    distance = 0
    homeDistance = 0
    awayDistance = 0
    for t in range(len(X_test)):
        print("TRUTH: %f %f" % (Y_test[t][0], Y_test[t][1]))
        prediction = lr.predict([X_test[t]])
        if (Y_test[t][0] > Y_test[t][1]):
            if (prediction[0][0] > prediction[0][1]):
                correct+=1
        else:
            if (prediction[0][0] < prediction[0][1]):
                correct+=1
        print("PREDICTION: %f %f" % (prediction[0][0], prediction[0][1]))
        distance += abs(float(prediction[0][0])-float(Y_test[t][0])) + abs(float(prediction[0][1])-float(Y_test[t][1]))
        homeDistance += abs(float(prediction[0][0])-float(Y_test[t][0]))
        awayDistance += abs(float(prediction[0][1])-float(Y_test[t][1]))
    print("Distance: %f" % (distance/float(X_test.shape[0])))
    print("HomeDistance: %f" % (homeDistance/float(X_test.shape[0])))
    print("AwayDistance: %f" % (awayDistance/float(X_test.shape[0])))
    print("Accuracy: %f" % (correct/float(len(X_test))))
    return

def dealWithLR():
    X = load_data("game_data_new.csv")
    Y = load_data("score_data_new.csv", True)
    X_train, X_test, Y_train, Y_test = no_shuffle_train_test_split(X,Y, 0.9)
    X_test = X_test[:331]
    Y_test = Y_test[:331]
    lr = linear_model.LinearRegression()
    lr.fit(X_train,Y_train)
    testResults(lr, X_train, Y_train)
def main():
    if (doSVM):
        dealWithSVM()
    if (doLinearRegression):
        dealWithLR()
    return

if __name__ == '__main__':
    main()
