# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:40:31 2019

@author: Administrator
"""


import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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

weights={
         'in':tf.Variable(tf.random_normal([input_size,features])),
         'out':tf.Variable(tf.random_normal([features,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[features,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }

def LoadTrainingData():
    raw_trainset = pd.read_csv(TrainingFile,engine='python')
    training_data = raw_trainset.iloc[:,0:input_size]
    training_label = raw_trainset.iloc[:-1]
    scaler=StandardScaler()
    scaler=scaler.fit(training_data)
    normalizedX=scaler.transform(training_data)
    training_label = training_label.values
    train_x, train_y = [],[]
    batch_index = []
    for i in range(len(normalizedX)-time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalizedX[i:i+time_step,:input_size]
        y = training_label[i:i+time_step,-1,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalizedX)-time_step))
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return batch_index, train_x, train_y


def LoadTestingData():
    original_data = pd.read_csv(TestingFile,engine='python')
    testing_data = original_data.iloc[:,0:input_size]
    testing_label = original_data.iloc[:,input_size]
    scaler=StandardScaler()
    scaler=scaler.fit(testing_data)
    standard_x=scaler.transform(testing_data)
    testing_label = testing_label.values
    div = (len(standard_x)+time_step-1)//time_step
    test_x, test_y = [],[]
    for i in range(div-1):
        x = standard_x[i*time_step:(i+1)*time_step,:input_size]
        y = testing_label[i*time_step:(i+1)*time_step,]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return test_x,test_y


## keras layers lstmcell rnn
def lstm(X):
        w_in=weights['in']
        b_in=biases['in']
        # 4 layers LSTM model
        input_raw=tf.reshape(X,[-1,input_size])
        # first layer---Dense layer :linear transformation
        input_rnn=tf.matmul(input_raw,w_in)+b_in
        input_rnn=tf.reshape(input_rnn,[-1,time_step,features])
        # second and third layer --LSTM layers
        cells=[tf.keras.layers.LSTMCell(features,dropout=0.05),
               tf.keras.layers.LSTMCell(features,dropout=0.05),]
        out_lstm,final_state,_=tf.keras.layers.RNN(cells,return_sequences=True,return_state=True)(input_rnn)
        #作为输出层的输入
        output=tf.reshape(out_lstm,[-1,features]) 
        output=tf.nn.leaky_relu(output)
        # fourth layer: linear transformation
        w_out = weights['out']
        b_out = biases['out']
        pred=tf.matmul(output,w_out)+b_out
        pred=tf.tanh(pred)
        return pred,final_state


def TrainLSTM(): 
    """
    set the optimizer and loss function, the we run this model to minimize the loss function
    """
    X = tf.placeholder(tf.float32,[None, time_step, input_size])
    Y = tf.placeholder(tf.float32,[None, time_step, output_size])
    batch_index,train_x,train_y = LoadTrainingData()
    train_x=train_x.astype(np.float32)
    train_y=train_y.astype(np.float32)
    lstm_out,_= lstm(X)
    Loss = tf.reduce_mean(tf.square(tf.reshape(lstm_out,[-1])-tf.reshape(Y,[-1])))
    optimizes = tf.train.AdamOptimizer(learning_rate).minimize(Loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    loss_value = []
    iteration = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1500):
            for j in range(len(batch_index)-1):
               # print(out_init)
                losses,_ = sess.run([Loss,optimizes],feed_dict={X:train_x[batch_index[j]:batch_index[j+1]],Y:train_y[batch_index[j]:batch_index[j+1]]})
            if(i%100==0):
                print("number of iteration:",i,"loss value is:",losses)
            iteration.append(i)
            loss_value.append(losses)
        print("model_save: ",saver.save(sess,r"E:\quant_trading\timing_strategy_of_50ETF index\model\lstm.ckpt"))
        plt.plot(iteration,loss_value,label='Loss Function')
        plt.legend()
        plt.show()
        print("Training Process Finished !")    
        
        


def LSTMPredict():
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    test_x, test_y = LoadTestingData()      
    test_x=test_x.astype(np.float32)
    test_y=test_y.astype(np.float32)
    output_data,_ = lstm(X)
    model_dir = r"E:\quant_trading\timing_strategy_of_50ETF index\model"
    saver = tf.train.Saver(tf.global_variables())
    #model_file = tf.train.latest_checkpoint(model_dir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        #saver.restore(sess,model_file) 
        yhat = []
        for i in range(len(test_x)):
            te=test_x[i].reshape(-1,time_step,input_size)
            p = sess.run(output_data,feed_dict={X:te})
            yhat.append(tf.reshape(p,[-1]))
        y = tf.reshape(test_y,[-1])
    return y,yhat


def PreAccuracy(y,yhat):
    y = tf.reshape(y,[1,-1])
    yhat = tf.reshape(yhat,[1,-1])
    with tf.Session() as sess:
        array_y = np.array(y.eval(session=sess))
        array_yhat = np.array(yhat.eval(session=sess))
    array_yhat = np.reshape(array_yhat,[-1,1])
    array_yad = np.reshape(array_y,[1,-1])
    array_yhat1 = []
    for i in range(np.shape(array_yhat)[0]):
        if array_yhat[i] > -0.15:
            array_yhat1.append(1)
        elif array_yhat[i] < -0.15:
            array_yhat1.append(-1)
    array_yhat1 = np.reshape(np.array(array_yhat1),[1,-1])
    count_array1 = array_yhat1 - array_yad
    total_accuracy = (np.sum(count_array1==0))/(np.shape(array_yad)[1])
   # accuracy of 1
    array_yhat2 = []
    for i in range(np.shape(array_yhat)[0]):
        if array_yhat[i] > -0.15:
            array_yhat2.append(1)
        else:
            array_yhat2.append(-100)
    array_yhat2 = np.reshape(np.array(array_yhat2),[1,-1]) 
    count_array2 = array_yhat2 - array_yad
    plus1_accuracy = (np.sum(count_array2==0))/(np.sum(array_yad==1))
    # accuracy of -1
    array_yhat3 = []
    for i in range(np.shape(array_yhat)[0]):
        if array_yhat[i] < -0.15:
            array_yhat3.append(-1)
        else:
            array_yhat3.append(100)
    array_yhat3 = np.reshape(np.array(array_yhat3),[1,-1])
    count_array3 = array_yhat3 - array_yad
    minus1_accuracy = (np.sum(count_array3==0))/(np.sum(array_yad==-1))
    return total_accuracy,plus1_accuracy,minus1_accuracy


if __name__ == "__main__":
  TrainLSTM()
  print("the training process is finished")
  print("Start Predict the process")
  y,yhat = LSTMPredict()
  accuracy1,acc2,acc3 = PreAccuracy(y,yhat)
  print('the accuracy of the lstm model is: '+str(accuracy1)+'%')
  print('the +1 accuracy is: '+str(acc2)+'%')
  print('the -1 accuracy is: '+str(acc3)+'%')






    
