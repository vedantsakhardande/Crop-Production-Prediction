import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv",usecols = ['Weather'])
df=data.values
X=df
data = pd.read_csv("data.csv",usecols = ['Ratio'])
df=data.values
y=df
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_
weather=float(input("Enter the temperature to predict the ratio :"))
print("The Precicted Value is ",reg.predict(weather))