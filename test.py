import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import datetime
import calendar
# import string

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
import scrapy
from scrapy.crawler import CrawlerProcess
# import BeautifulSoup
import os
import sys
import pyttsx3
# from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import time
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor

# scrapy api imports
from scrapy import signals, log
from scrapy.crawler import Crawler
from scrapy.settings import Settings
# from multiprocessing import Process, Queue

RUNNING_CRAWLERS = []
class QuotesSpider(scrapy.Spider):
    def __init__(self):
        self.result=""    
    now = datetime.datetime.now()
    year=now.year
    month=now.month
    day=now.day
    one=year%10
    print(one)
    year=year/10
    two=int(year%10)
    print(two)
    yr=two*10+one
    print("Day ",day,"Month ",month,"Year ",yr)
    monthname=calendar.month_name[(month+1)%12]
    # "https://www.accuweather.com/en/in/ratnagiri/189289/"+str(monthname.lower())+"-weather/189289?monyr="+str((month+1)%12)+"/"+str(day)+"/"+str(yr)+"&view=table"
    start_urls = [
                    "https://www.accuweather.com/en/in/ratnagiri/189289/"+str(monthname.lower())+"-weather/189289?monyr="+str((month+1)%12)+"/"+str(day)+"/"+str(yr)+"&view=table"
                ]
    def parse(self, response):
            
            now = datetime.datetime.now()
            year=now.year
            month=now.month
            day=now.day
            one=year%10
            print(one)
            year=year/10
            two=int(year%10)
            print(two)
            yr=two*10+one
            print("Day ",day,"Month ",month,"Year ",yr)
            monthname=calendar.month_name[(month+1)%12]
            # "https://www.accuweather.com/en/in/ratnagiri/189289/"+str(monthname.lower())+"-weather/189289?monyr="+str((month+1)%12)+"/"+str(day)+"/"+str(yr)+"&view=table"
            start_urls = [
                    "https://www.accuweather.com/en/in/ratnagiri/189289/"+str(monthname.lower())+"-weather/189289?monyr="+str((month+1)%12)+"/"+str(day)+"/"+str(yr)+"&view=table"
                ]
            print(start_urls)
            for row in response.xpath('//*[@class="table table-striped"]//tbody//tr'):
                answer= {
                    'temp' : row.xpath('td[1]//text()').getall(),
                    # 'link': response.css("div.g a::attr(href)").extract(),          
                }
            print("ANSWER :",answer)
            f=open("output.txt","w")
            f.write(answer)
            f.close()
            answerlist=answer["text"]
            link=answer["link"]
            linkstr=link[0]
            linkstrpart=linkstr.split("=")[1]
            finallink=linkstrpart.split("&")[0]
            result=""
            flag=0
            for i in range(0,len(answerlist)):
                if('.' in answerlist[i]):
                    flag=1
                    index=i
                if(flag==1):
                    break
            for i in range(1,index+1):
                result=result+answerlist[i]
            result=result+"For More Information Refer the following"
            engine = pyttsx3.init()
            engine.say(result)
            engine.runAndWait()
            result=result+"\n"+"Link : "+finallink
            f = open("data.txt", "w")
            f.write(result)
            f.close()
            self.result=result

app = Flask(__name__)
CORS(app)
basedir = os.path.abspath(os.path.dirname(__file__))
# app.config['SQLALCHEMY_DATABASE_URI'] = 'jdbc:mysql://localhost:3306/charity'
# db = SQLAlchemy(app)
# ma = Marshmallow(app)

# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     sender = db.Column(db.String(120), unique=True)
#     receiver = db.Column(db.String(120), unique=True)

#     def __init__(self, username, email):
#         self.username = username
#         self.email = email


# class UserSchema(ma.Schema):
#     class Meta:
#         # Fields to expose
#         fields = ('username', 'email')


# user_schema = UserSchema()
# users_schema = UserSchema(many=True)

def get_answer(search):
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })
    process.crawl(QuotesSpider)
    process.start()
    RUNNING_CRAWLERS.append("QuotesSpider")
    # with open ("data.txt", "r") as myfile:
    #     data=myfile.readlines()
    # myfile.close()
    # finalans=""
    # for i in range(0,len(data)):
    #     finalans=finalans+data[i]
    # print(finalans)
    # return finalans

app.run(debug=True)


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
def get_test():
    data = pd.read_csv("test.csv",usecols = ['Avg Month Temp'])
    df=data.values
    X=df
    data = pd.read_csv("test.csv",usecols = ['Ratio'])
    df=data.values
    y=df
    # print("Y NEW SHAPE :",y.shape)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y = np.reshape(y, (y.shape[0]))
    # print("Y NEW CHANGED SHAPE :",y.shape)
    # print("X NEW :",X)
    # print("Y NEW :",y)
    return X, y



now = datetime.datetime.now()
year=now.year
month=now.month
day=now.day
one=year%10
print(one)
year=year/10
two=int(year%10)
print(two)
yr=two*10+one
print("Day ",day,"Month ",month,"Year ",yr)
monthname=calendar.month_name[(month+1)%12]
print("https://www.accuweather.com/en/in/ratnagiri/189289/"+str(monthname.lower())+"-weather/189289?monyr="+str((month+1)%12)+"/"+str(day)+"/"+str(yr)+"&view=table")
model = load_model('lstm_model.h5')
X,ytrain=get_train()
X,ytest = get_test()
# make predictions
weather=float(input("Enter the temperature to predict the ratio :"))
# x=np.empty((), dtype=float, order='C')
# x=np.array([[[weather]]],np.float32)
# print(x)
print("X :",X)
area=int(input("Enter the value of area used for farming :"))
X=np.array([[[weather]]],np.float32)
ytest = model.predict(X, verbose=0)
print("Predicted Output is :")
print(ytest)
yvalue=ytest[0]
ratio=yvalue[0]
print("The Possible production of Rice is :",area*ratio)

# matrix=confusion_matrix(ytest.argmax(axis=0),ytrain.argmax(axis=0))
# print(matrix)
# scores=model.evaluate(X,ytest)
# print("Test Accuracy :",scores)