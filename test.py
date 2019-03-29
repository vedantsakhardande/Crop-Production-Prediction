import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model

model = load_model('lstm_model.h5')
# make predictions
weather=float(input("Enter the temperature to predict the ratio :"))
# x=np.empty((), dtype=float, order='C')
x=np.array([[[weather]]],np.float32)
print(x)

yhat = model.predict(x, verbose=0)
print("Predicted Output is :")
print(yhat)