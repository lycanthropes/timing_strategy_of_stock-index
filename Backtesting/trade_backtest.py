# -*- coding: utf-8 -*-
"""
author: Wang
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
import math
from matplotlib.ticker import MultipleLocator
import os 
os.chdir(r'E:\quant_trading\timing_strategy_of_50ETF index\Model_Train')
import predict_VIX_fluctuation

time_step = 5
#reload(sys)
#sys.setdefaultencoding('utf-8')

####### Time&&Data
features = r"E:\quant_trading\timing_strategy_of_50ETF index\Data\feature.csv"
TestingFile = r"E:\quant_trading\timing_strategy_of_50ETF index\Data\test.csv"
rural_data = pd.read_csv(TestingFile)
fea_data = pd.read_csv(features)
date = fea_data.iloc[:,-1]
OpenPrice = rural_data.iloc[:,0]
ClosePrice = rural_data.iloc[:,1]


####### Set the trading flag/account
flag = 0
shut = 0
account = 50000
fee = 10.0
annualizedFactor = 365/219

accumulation = 0
occupation = []
R = []
time = []

cost = 0
rev = 0
yhat = predict_VIX_fluctuation.Predict()
####### Begin the BackTest
for i in np.arange(142):
    prediction=yhat[i,4,:]
    prediction=round(prediction[0],0)
    #################PART1 BUY SIDE#################
    ########## 一次买入1000股 && 当日平仓
    ###### Open Buy
    if prediction == 1:
        if flag == 0:
            buy_price = OpenPrice[i]
            print(str(date[i])+" Buy 50ETF"+" the price is "+str(buy_price))
            accumulation += buy_price*1000
            flag = 1
            cost = buy_price * 1000 + fee
            occupation.append(accumulation)
    ###### Close Sell
        elif flag != 0:
            sell_price = ClosePrice[i]
            print(str(date[i]) + " Sell 50ETF"+" the price is "+str(sell_price))
            accumulation -= sell_price*1000
            flag = 0
            rev = sell_price * 1000 - fee
            R.append(rev-cost)
            #### 平仓
            #cost = 0
            time.append(date[i])
            occupation.append(accumulation)
    #################PART2 SELL SIDE##################
    #########一次卖出1000股 && 当日平仓
    ###### Open Sell
    elif prediction == -1:
        if flag == 0:
            sell_price = OpenPrice[i]
            print(str(date[i])+" Sell 50ETF"+" the price is"+str(sell_price))
            accumulation -= sell_price*1000
            flag = -1
            rev = sell_price*1000 - fee
            occupation.append(accumulation)
    ####### Close Buy
        elif flag != 0:
            buy_price = ClosePrice[i]
            print(str(date[i])+" Buy 50ETF"+" the price is"+str(buy_price))
            accumulation += buy_price*1000
            flag = 0
            cost = buy_price*1000 + fee
            R.append(rev-cost)
            time.append(date[i])
            occupation.append(accumulation)
        
    
#######计算交易评估数据
sumReturn = sum(R)/account
occupation = np.array(occupation)
maxOccupateRate = np.max(occupation)/account
annualReturn = round(sumReturn*annualizedFactor,7)
rf=0.04
annual_sharpe_Ratio=(annualReturn-rf)/(np.std(R,ddof=1)*np.sqrt(len(R)/2.83))
annual_volatility=(annualReturn-rf)/annual_sharpe_Ratio


cum_Return = []
sum = 0
for i in R:
    sum += i
    cum_Return.append(sum)

money = []
for k in cum_Return:
    money.append(50000+k)


max_drawdown =0
for e, i in enumerate(money):
    for f, j in enumerate(money):
        if f > e and float(j - i)  < max_drawdown:
            max_drawdown = float(j - i)

max_drawdownratio =0
try:
    for e, i in enumerate(money):
        for f, j in enumerate(money):
            if f > e and float((j - i)/i)  < max_drawdownratio:
                max_drawdownratio = float((j - i)/i)
except:
    max_drawdownratio=None


win = 0
for i in R:
    if i > 0:
        win += 1
        
        
win_rate = win/float(len(R))


plt.figure(figsize=(10, 5))
#summary
print('Return')
print(R)
print('win_rate','annualReturn','annual_sharpe_Ratio','annual_volatility','max_drawdown','max_drawdownratio','maxOccupateRate')
print(win_rate,annualReturn,annual_sharpe_Ratio,annual_volatility,max_drawdown,max_drawdownratio,maxOccupateRate)

plt.plot(time, money)
plt.xlabel('Date')
plt.ylabel('Money')
plt.title('Money Curve')
plt.grid(True)
plt.show()
