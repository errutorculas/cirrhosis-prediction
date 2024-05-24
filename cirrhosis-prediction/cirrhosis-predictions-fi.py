import warnings
import itertools
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from keras.layers import Conv1D, Dense, Input, MaxPooling1D, Dropout, Flatten
from keras.models import Sequential


df = pd.read_csv('input/cirrhosis.csv')
df = df.drop('ID', axis=1)


""" Handling Missing Values """

# I'm going to drop the 6 rows with missing 'Stage'
df = df[df['Stage'].notna()]

# Numerical --> Median
numerical_columns = df.select_dtypes(include=(['int64', 'float64'])).columns

for c in numerical_columns:
    df[c].fillna(df[c].median(), inplace=True)
    
# Categorical --> Most Frequent
categorical_columns = df.select_dtypes(include=('object')).columns

for c in categorical_columns:
    df[c].fillna(df[c].mode().values[0], inplace=True)

    
df.Stage = df.Stage.astype(int)


# Dummy Variables
df['Sex'] = df['Sex'].replace({'M':0, 'F':1})
df['Ascites'] = df['Ascites'].replace({'N':0, 'Y':1})
df['Drug'] = df['Drug'].replace({'D-penicillamine':0, 'Placebo':1})
df['Hepatomegaly'] = df['Hepatomegaly'].replace({'N':0, 'Y':1})
df['Spiders'] = df['Spiders'].replace({'N':0, 'Y':1})
df['Edema'] = df['Edema'].replace({'N':0, 'Y':1, 'S':-1})
df['Status'] = df['Status'].replace({'C':0, 'CL':1, 'D':-1})


X = df.drop(['Status', 'N_Days', 'Stage'], axis=1)
y = df.pop('Stage')


# Upsamlping
sm = SMOTE(k_neighbors = 3)
X, y = sm.fit_resample(X, y)


# Scaling Data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)

# In[]

RFmodel = RandomForestClassifier()
RFmodel.fit(X_train, y_train)

# In[]
model_mlp = Sequential()

input_shape = (X_train.shape[1],)

model_mlp.add(Dense(32, activation='relu', input_shape=(input_shape)))
model_mlp.add(Dense(64, activation='relu'))
model_mlp.add(Dense(32, activation='relu'))
model_mlp.add(Dense(5, activation='softmax'))

model_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_mlp.build(input_shape=(X_train.shape))

h_mlp = model_mlp.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)


# In[]:

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

h_cnn = model_cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)


# dataPoint = pd.read_csv('input/for-predictions.csv')
dataPoint = pd.read_csv(sys.argv[1])

dataPointDict = pd.DataFrame({"Drug": [1], 
                          "Age": [14060],
                          "Sex": [1],
                          "Ascites": [0],
                          "Hepatomegaly": [0],
                          "Spiders": [0],
                          "Edema": [0],
                          "Bilirubin": [0.7],
                          "Cholesterol": [0],
                          "Albumin": [0],
                          "Copper": [0],
                          "Alk_Phos": [0],
                          "SGOT": [49.6],
                          "Tryglicerides": [56],
                          "Platelets": [265],
                          "Prothrombin": [11]})


# In[]:

RandomForest_predict = RFmodel.predict(dataPoint)

print(f" [RF] Cirrhosis Predicted Stage: {RandomForest_predict[0]}")



# In[62]:

MLP_predict = np.argmax(model_mlp.predict(dataPoint), axis=-1)

print(f"[MLP] Cirrhosis Predicted Stage: {MLP_predict[0]}")


# In[63]:

CNN_predict = np.argmax(model_cnn.predict(dataPoint), axis=-1)

print(f"[CNN] Cirrhosis Predicted Stage: {CNN_predict[0]}")

