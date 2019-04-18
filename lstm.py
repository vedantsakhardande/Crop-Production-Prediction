import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model

# return training data
def get_train():
    data = pd.read_csv("train.csv",usecols = ['Avg Month Temp'])
    df=data.values
    X=df
    data = pd.read_csv("train.csv",usecols = ['Ratio'])
    df=data.values
    y=df
    # print("Y NEW SHAPE :",y.shape)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y = np.reshape(y, (y.shape[0]))
    # print("Y NEW CHANGED SHAPE :",y.shape)
    # print("X NEW :",X)
    # print("Y NEW :",y)
    return X, y

# define model
model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
# model.add(Dense(1, activation='linear'))
model.add(Dense(1, activation='relu'))
# compile model
model.compile(loss='mse', optimizer='adam')
# fit model
X,y = get_train()
model.fit(X, y, epochs=300, shuffle=False, verbose=0)
# save model to single file
model.save('lstm_model.h5')

# snip...
# later, perhaps run from another script

# load model from single file
# model = load_model('lstm_model.h5')
# # make predictions
# weather=float(input("Enter the temperature to predict the ratio :"))
# # x=np.empty((), dtype=float, order='C')
# x=np.array([[[weather]]],np.float32)
# print(x)

# yhat = model.predict(x, verbose=0)
# print("Predicted Output is :")
# print(yhat)