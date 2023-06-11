import sys

from sklearn import svm
import pandas as pd
import numpy as np

import pickle
import os

#disctionary for mapping letters to numbers
letters = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": 5,
        "g": 6,
        "h": 7,
        "i": 8,
        "k": 9,
        "l": 10,
        "m": 11,
        "n": 12,
        "o": 13,
        "p": 14,
        "q": 15,
        "r": 16,
        "s": 17,
        "t": 18,
        "u": 19,
        "v": 20,
        "w": 21,
        "x": 22,
        "y": 23,
    }
reverse_letters = {value: key for key, value in letters.items()}

def load_data(path, temp):
    # dropping 0.3% of outliers of given data
    if temp == True:
        data = pd.read_csv(path)
        # dropping columns with world_landmark
        data = data[[col for col in data.columns if not col.startswith("world_landmark_")]]
        data["letter"] = data["letter"].map(letters)
        # dropping first column and handedness.score
        data = data.drop(columns = [data.columns[0], "handedness.score", "handedness.label"])
        for col in data.columns:
            lower_percentile = data[col].quantile(0.003)
            upper_percentile = data[col].quantile(0.997)
            for i in range(len(data[col])):
                if data[col][i] < lower_percentile:
                    data.loc[i] = np.nan
                elif data[col][i] > upper_percentile:
                    data.loc[i] = np.nan
        data = data.dropna()
    #splitting data into X and y
        X = data.drop(["letter"], axis=1)
        y = data["letter"]

        X = np.array(X)
        y = np.array(y)
        return X, y
    else:
        data = path
        # dropping columns with world_landmark
        data = data[[col for col in data.columns if not col.startswith("world_landmark_")]]
        # mapping letters and hands to numbers
        data = data.drop(columns = [data.columns[0], "handedness"])
        # data["handedness"] = data["handedness"].map(arm)
        return data

def perform_processing(data: pd.DataFrame) -> pd.DataFrame:
    # saving/loading model to/from file
    model_path = 'finalized_model.sav'
    if os.path.exists(model_path):
        # loading model from file
        clf = pickle.load(open(model_path, 'rb'))
    else:
        # loading data for training
        X, y = load_data("all_data.csv", True)
        # creating model
        clf = svm.SVC(kernel="linear", C=170).fit(X, y)
        # saving model to file
        pickle.dump(clf, open(model_path, 'wb'))

    #testing data
    X_ext = load_data(data, False)
    pred = clf.predict(X_ext)

    # reverse mapping numbers to letters
    predicted_letters = [reverse_letters[num] for num in pred]
    predicted_data = pd.DataFrame(predicted_letters, columns=['letter'])

    return predicted_data