import sys

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
# import time

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
#disctionary for mapping left and right hand to numbers
arm = {"Left": 0, "Right": 1}
#function for loading data from csv file
def load_data(path, temp):
    data = pd.read_csv(path)
    #dropping columns with world_landmark
    data = data[[col for col in data.columns if not col.startswith("world_landmark_")]]
    #mapping letters and hands to numbers
    data["handedness.label"] = data["handedness.label"].map(arm)
    data["letter"] = data["letter"].map(letters)
    #dropping first column and handedness.score
    data = data.drop([data.columns[0], "handedness.score"], axis=1)
    # dropping 0.3% of outliers of given data
    if temp == True:
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

#loading data for training
X, y = load_data("all_data.csv", True)

#splitting data into train and test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.12, random_state=42, stratify=y
# )

#creating model
# print(X.shape)
# start_time = time.time()
clf = svm.SVC(kernel="linear", C=2000).fit(X, y)
# clf = LinearSVC(C=2000).fit(X_train, y_train)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Czas wykonania: {execution_time} sekund")

#saving/loading model to/from file
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
# clf = pickle.load(open(filename, 'rb'))

#loading testing data
path1 = sys.argv[1]
# path1 = "Dane_ADAM.csv"
X_ext, y_ext = load_data(path1, False)
pred = clf.predict(X_ext)
# print("Score:",(clf.score(X_ext, y_ext))*100,"%")

#reverse mapping numbers to letters
predicted_letters = [reverse_letters[num] for num in pred]
df = pd.DataFrame(predicted_letters)

#saving predicted letters to csv file
path2 = sys.argv[2] + 'dane.csv'
# path2 = 'dane.csv'
df.to_csv(path2, index=False)