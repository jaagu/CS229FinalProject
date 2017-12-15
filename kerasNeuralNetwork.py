import getData
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import pickle
import os

FOLDERNAME = "TWOLAYERKERAS_"
LOADING_PRETRAINED_MODEL = False

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

def average_distance(nn, x_test,y_test, filename, iteration = 0, write_scores = False):
    distance = 0
    correct = 0 
    for i in range(x_test.shape[0]):
        test = x_test[i].reshape(1,-1)
        prediction = nn.predict(test)
        truth = y_test[i]
        if write_scores:
            with open("%sdistances/afterTestscores_%s_%i.txt" % (FOLDERNAME, filename,iteration), 'a') as f:
                f.write("%i %i %f %f\n" % (truth[0], truth[1], prediction[0][0], prediction[0][1]))
        if (((prediction[0][0] > prediction[0][1]) and (truth[0] > truth[1])) or ((prediction[0][0] < prediction[0][1]) and (truth[0] < truth[1]))):
            correct += 1
        distance += abs(prediction[0][0]-truth[0]) + abs(prediction[0][1]-truth[1])
    if (write_scores == False):
        avg_accuracy = correct/float(x_test.shape[0])
        avg_distance = distance/x_test.shape[0]
        print("Average %s distance= %f" % (filename,avg_distance))
        print("Average %s accuracy= %f" % (filename,avg_accuracy))
        with open("%saccuracy/%s_accuracy.txt" % (FOLDERNAME, filename), 'a') as f:
            f.write("%s\n" % str(avg_accuracy))
        with open("%sdistances/%s_distance.txt" % (FOLDERNAME, filename), 'a') as f:
            f.write("%s\n" % str(avg_distance))

##    print('\n')

def larger_model():
    model = Sequential()
    model.add(Dense(input_dim=60, units=60, kernel_initializer='random_normal',activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=30, kernel_initializer='random_normal',activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=16, kernel_initializer='random_normal',activation='relu'))
    model.add(Dense(units=2, kernel_initializer='random_normal'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model

def main():
    X = load_data("game_data_new.csv")
    Y = load_data("score_data_new.csv", True)
    X_train, X_test, Y_train, Y_test = no_shuffle_train_test_split(X,Y, 0.9)
    X_test = X_test[:331]
    Y_test = Y_test[:331]
    with open("%sdistances/test_distance.txt" % FOLDERNAME, 'w') as f:
        f.write("Distances\n")
    with open("%sdistances/train_distance.txt" % FOLDERNAME, 'w') as f:
        f.write("Distances\n")
    with open("%saccuracy/test_accuracy.txt" % FOLDERNAME, 'w') as f:
        f.write("Accuracies\n")
    with open("%saccuracy/train_accuracy.txt" % FOLDERNAME, 'w') as f:
        f.write("Accuracies\n")
    if(LOADING_PRETRAINED_MODEL):
        lst = sorted(os.listdir("TWOLAYERKERAS_models")[1:], key=lambda x:int(x.split('.')[1]))
        for filename in lst:
            if filename.endswith(".hdf5"):
                print(filename)
                model = larger_model()
                model.load_weights("TWOLAYERKERAS_models/%s" % filename)
                average_distance(model, X_test, Y_test, "aftertest", int(filename.split('.')[1]), False)
        
    else:
        filepath = "TWOLAYERKERAS_models/weights.{epoch:02d}.hdf5"
        print(filepath)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose =1, save_best_only=True, mode='auto', period = 10)
        callbacks_list = [checkpoint]
        NUM_EPOCHS = 100000
        epochCount = 0
        for i in range(100000000):
            model = larger_model()
            history = model.fit(X_train,Y_train,validation_split=0.1, epochs = NUM_EPOCHS, batch_size = 100,
                      callbacks=callbacks_list, verbose = 0, initial_epoch = epochCount)
            epochCount +=NUM_EPOCHS
            pyplot.plot(history.history['mean_squared_error'])
            pyplot.show()
    return

if __name__ == '__main__':
    main()
