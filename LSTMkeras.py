from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

# LSTM模型的参数设置
# 每个隐藏层的节点数目
features = 15   
# 学习率
learning_rate = 0.01
# 输入数据的dimension number,即列数
input_size = 7
#输出ｙ的dimension number 即为单值输出（-1或1)。
output_size = 1
# mini-batch gradientdescent的batch
batch_size = 10
# 滞后期数，相当于rolling window的宽度
time_step = 5
# 训练集path
TrainingFile = r"E:\quant_trading\timing_strategy_of_50ETF index\Data\train.csv"
# 测试集path
TestingFile = r"E:\quant_trading\timing_strategy_of_50ETF index\Data\test.csv"
#　取出训练集
xy = np.loadtxt(TrainingFile, delimiter=',',skiprows=1)
# 以前n-1列数据作为x
x = xy[:,0:-1]
# 以最后一列数据作为y
y = xy[:, [-1]]


dataX = []
dataY = []
# rolling window, 将二维数据x,y扩展为rolling的三维数据，shape为[sample_numbers,time_step,input_size]
for i in range(0, len(y) - time_step):
    _x = x[i:i + time_step]
    _y = y[i + time_step]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)
# 将其转换为np.array类型
trainX= np.array(dataX)
trainY = np.array(dataY)
# 生成四层的模型，并对训练集数据进行拟合
    """
    the structure of the lstm: the input layer --->activate function 1---->Multi-RNN---->activate function 2
    --->output layer--->activate function 3--->the prediction of label
    any lstm layer has its own dopout 
   """
#定义序列模型
model=Sequential()
# first layer --Dense layer，由于为3维的时间序列数据，采用TimeDistributed包裹
model.add(tf.keras.layers.TimeDistributed(Dense(features,kernel_initializer='RandomNormal',activation=tf.nn.leaky_relu,input_shape=(time_step,input_size))))
# second layer--LSTM , activation=default (tanh)
model.add(LSTM(features,return_sequences=True))
# add dropout
model.add(Dropout(0.03))
#Third layer-- LSTM layer, activation=leaky_relu
model.add(LSTM(features,activation=tf.nn.leaky_relu))
# Fourth layer-- Dense layer
model.add(Dense(output_size))
model.add(Activation('tanh'))
adms=Adam(lr=learning_rate)
model.compile(loss='mse',optimizer=adms,metrics=['mse'])
model.fit(trainX,trainY,batch_size=batch_size,epochs=500)


# 打印并输出loss散点图数据
history=model.fit(train_x,train_y,epochs=1000,verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

############ 对模型在测试集上进行检验
test_data = np.loadtxt(TestingFile, delimiter=',',skiprows=1)
# 以前n-1列数据作为x
test_x = test_data[:,0:-1]   
# 以最后一列数据作为y
test_y = test_data[:, [-1]]


tdX = []
tdY = []
# rolling window, 将二维数据x,y扩展为rolling的三维数据，shape为[sample_numbers,time_step,input_size]
for i in range(0, len(test_y) - time_step):
    _tx = test_x[i:i + time_step]
    _ty = test_y[i + time_step]  # Next close price
    print(_tx, "->", _ty)
    tdX.append(_tx)
    tdY.append(_ty)
# 将其转换为np.array类型
ttestX= np.array(tdX)
ttestY = np.array(tdY)
# 评价模型在测试集的精确度和loss
loss, accuracy = model.evaluate(ttestX, ttestY)

# 预测未来数据---对提供的X数据进行预测
predictions = model.predict(X)
