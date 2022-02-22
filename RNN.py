# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:27:02 2021

@author: saurabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path="C:\\Users\\saurabh\\Desktop\\Internships\\Deep Learning\\Churn_Modelling.csv"
dataset_train = pd.read_csv(path)

training_set = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc  =MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train  =[]
y_train =[]

for i in range(50,1258):
    x_train.append(training_set_scaled[i-50:i,0])
    y_train.append(training_set_scaled[i,0])
    

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train.shape[1]

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#implementing RNN Model now
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()
model.add(LSTM(units = 64, return_sequences = True, input_shape = (x_train.shape[1],1)))

model.add(LSTM(units = 64, return_sequences = True))

model.add(LSTM(units = 64))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error)

model.fit(x_train,y_train, epochs =100, batch_size=16)