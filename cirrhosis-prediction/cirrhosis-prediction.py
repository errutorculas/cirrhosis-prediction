# pip install ...

import warnings
warnings.filterwarnings('ignore')
import itertools

import pandas as pd
import numpy as np
import tensorflow as tf
import sys, argparse          # command line arguments
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import SMOTE

from keras.layers import Conv1D, Dense, Input, MaxPooling1D, Dropout, Flatten
from keras.models import Sequential

parser=argparse.ArgumentParser()

parser.add_argument("--model", '-m', help="Model Train (RF, SVM, MLP, CNN)")
parser.add_argument("--predict", '-p', help="Predict datapoint")

args = parser.parse_args()


df = pd.read_csv('input/cirrhosis.csv')

df = df.drop('ID', axis=1)

""" Handling Missing Values

    Dropping rows with missing values
"""
df = df[df['Stage'].notna()]                                                    # Drop rows with missing 'Stage'

numerical_columns = df.select_dtypes(include=(['int64', 'float64'])).columns    # Numerical --> Median

for c in numerical_columns:
    df[c].fillna(df[c].median(), inplace=True)
    
categorical_columns = df.select_dtypes(include=('object')).columns              # Categorical --> Most Frequent

for c in categorical_columns:
    df[c].fillna(df[c].mode().values[0], inplace=True)

df.Stage = df.Stage.astype(int)

""" Recoding and Mutating Data

    Translating the categorical values to a numerical value
"""
df['Sex'] = df['Sex'].replace({'M':0, 'F':1})
df['Ascites'] = df['Ascites'].replace({'N':0, 'Y':1})
df['Drug'] = df['Drug'].replace({'D-penicillamine':0, 'Placebo':1})
df['Hepatomegaly'] = df['Hepatomegaly'].replace({'N':0, 'Y':1})
df['Spiders'] = df['Spiders'].replace({'N':0, 'Y':1})
df['Edema'] = df['Edema'].replace({'N':0, 'Y':1, 'S':-1})
df['Status'] = df['Status'].replace({'C':0, 'CL':1, 'D':-1})


X = df.drop(['Status', 'N_Days', 'Stage'], axis=1)      # input variables
y = df.pop('Stage')                                     # output variable (Detecting the stage of Cirrhosis)


# Upsamlping
sm = SMOTE(k_neighbors = 3)
X, y = sm.fit_resample(X, y)

# Scaling Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)


""" Modeling the Cirrhosis prediction

    Random Forest, SVM, MLP, and CNN
"""

def modelTrainer(X_train, y_train, modelName, predictDataPoint):
    models= {"SVM": SVC(),
             "RF": RandomForestClassifier(),
             "DT": DecisionTreeClassifier(),
             "LR": LogisticRegression(max_iter=200),
             "Adaboost": AdaBoostClassifier(),
             "KNN": KNeighborsClassifier()}
    
    model = models[modelName]
    model.fit(X_train, y_train)

    dataPoint = pd.read_csv(predictDataPoint)

    model_predict = model.predict(dataPoint)

    print(f"\n[{modelName}] Cirrhosis Predicted Stage: {model_predict[0]}\n")

def modelTrainerMLP(X_train, y_train, X_test, y_test, predictDataPoint):
    model_mlp = Sequential()

    input_shape = (X_train.shape[1],)

    model_mlp.add(Dense(32, activation='relu', input_shape=(input_shape)))
    model_mlp.add(Dense(64, activation='relu'))
    model_mlp.add(Dense(32, activation='relu'))
    model_mlp.add(Dense(5, activation='softmax'))

    model_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_mlp.build(input_shape=(X_train.shape))
    model_mlp.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

    dataPoint = pd.read_csv(predictDataPoint)
    MLPModel_predict = model_mlp.predict(dataPoint)
    MLP_predict = np.argmax(MLPModel_predict, axis=1)
    print(f"\n[MLP] Cirrhosis Predicted Stage: {MLP_predict[0]}\n")


def modelTrainerCNN(X_train, y_train, X_test, y_test, predictDataPoint):
    model_cnn = Sequential()

    model_cnn.add(Conv1D(32, kernel_size=3, padding='same', activation='relu', input_shape=(16, 1)))
    model_cnn.add(MaxPooling1D(pool_size=2))

    model_cnn.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=2))

    model_cnn.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=2))

    model_cnn.add(Dropout(0.25))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(16, activation='relu'))

    model_cnn.add(Dense(5, activation='softmax'))

    model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')

    model_cnn.build(input_shape=(1, 16))

    X_train = np.expand_dims(X_train, axis=2)
    X_test =  np.expand_dims(X_test, axis=2)

    model_cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)


    dataPoint = pd.read_csv(predictDataPoint)
    CNNModel_predict = model_cnn.predict(dataPoint)
    CNN_predict = np.argmax(CNNModel_predict, axis=-1)
    print(f"\n[CNN] Cirrhosis Predicted Stage: {CNN_predict[0]}\n")



if args.model == 'RF':
    modelTrainer(X_train, y_train, 'RF', args.predict)

elif args.model == 'SVM':
    modelTrainer(X_train, y_train, 'SVM', args.predict)

elif args.model == 'KNN':
    modelTrainer(X_train, y_train, 'KNN', args.predict)

elif args.model == 'DT':
    modelTrainer(X_train, y_train, 'DT', args.predict)

elif args.model == 'LR':
    modelTrainer(X_train, y_train, 'LR', args.predict)

elif args.model == 'Adaboost':
    modelTrainer(X_train, y_train, 'Adaboost', args.predict)

elif args.model == 'MLP':
    modelTrainerMLP(X_train, y_train, X_test, y_test, args.predict)

elif args.model == 'CNN':
    modelTrainerCNN(X_train, y_train, X_test, y_test, args.predict)

elif args.model == 'all':
    modelTrainer(X_train, y_train, 'RF', args.predict)
    modelTrainer(X_train, y_train, 'SVM', args.predict)
    modelTrainerMLP(X_train, y_train, X_test, y_test, args.predict)
    modelTrainerCNN(X_train, y_train, X_test, y_test, args.predict)