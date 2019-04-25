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
from sklearn.preprocessing import StandardScaler

# LSTM模型的参数设置
# 每个隐藏层的节点数目
features = 15   
# 学习率
learning_rate = 0.005
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

def LoadData(file):
    xy = np.loadtxt(file, delimiter=',',skiprows=1)
    x=xy[:,0:-1]
    y = xy[:, [-1]]
    scaler=StandardScaler()
    scaler=scaler.fit(x)
    normalizedX=scaler.transform(x)
    train_x,train_y=[],[]
    for i in range(len(normalizedX)-time_step):
        _x=normalizedX[i:i + time_step]
        _y=y[i:i + time_step]
        train_x.append(_x)
        train_y.append(_y)
    train_x=np.array(train_x)
    train_y=np.array(train_y)
    return train_x,train_y


# 生成四层的模型，并对训练集数据进行拟合
    """
    the structure of the lstm: the input layer --->activate function 1---->Multi-RNN---->activate function 2
    --->output layer--->activate function 3--->the prediction of label
    any lstm layer has its own dopout 
   """
train_x,train_y=LoadData(TrainingFile)
#定义序列模型
model=Sequential()
# first layer --Dense layer，由于为3维的时间序列数据，采用TimeDistributed包裹
model.add(tf.keras.layers.TimeDistributed(Dense(features,kernel_initializer='RandomNormal',activation=tf.nn.leaky_relu,input_shape=(time_step,input_size))))
# second layer--LSTM , activation=default (tanh)
model.add(LSTM(features,return_sequences=True))
# add dropout
model.add(Dropout(0.03))
#Third layer-- LSTM layer, activation=leaky_relu
model.add(LSTM(features,return_sequences=True,activation=tf.nn.leaky_relu))
# Fourth layer-- Dense layer
model.add(Dense(output_size))
model.add(Activation('tanh'))
adms=Adam(lr=learning_rate)
model.compile(loss='mse',optimizer=adms,metrics=['mse'])
model.fit(train_x,train_y,batch_size=batch_size,epochs=500)


# 打印并输出loss散点图数据
history=model.fit(train_x,train_y,epochs=1000,verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("result of keras_model.png")
plt.show()


############ 对模型在测试集上进行检验
test_x,test_y=LoadData(TestingFile)
# 评价模型在测试集的精确度和loss
loss, accuracy = model.evaluate(test_x, test_y)


# 预测未来数据---对提供的X数据进行预测,下面一行代码，需要提供X
#predictions = model.predict(X)
