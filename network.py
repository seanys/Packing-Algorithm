'''
网络训练情况
1. 标准化：形状全部转化为向量表示，收缩将1000作为收缩参数，后续获得其最大值，宽度为其
2. 宽度：暂时不考虑宽度输入，默认为1500，后续"宽度/10000"作为第一个输入参数
3. 输出：分别为重心的x/y坐标
4. 旋转角度：后续增加旋转角度的处理
5. 利用率：可以预测利用率的范围
6. 可行化：(a) 采用Minimize Overlap，形状全部向下平移 (b) 采用最低高度进行序列生成
'''
from keras.models import Sequential,load_model
from keras import models,layers,optimizers
from keras.layers import Dense, Activation,Embedding,LSTM,Dropout
from shapely.geometry import Polygon
from tools.polygon import GeoFunc,PltFunc
import matplotlib.pyplot as plt
import random
import pandas as pd
import time
import csv
import numpy as np
import json

class DataLoad(object):
    def getLSTMData(self):
        x_train,y_train,x_val,y_val=[],[],[],[]
        _input=pd.read_csv("/Users/sean/Documents/Projects/Data/input.csv")
        _output=pd.read_csv("/Users/sean/Documents/Projects/Data/output_position.csv")

    def getPointerData(self):
        pass


class BPNetwork(object):
    def trainModel():
        pass

class LSTMPredict(object):
    def __init__(self):
        pass
    
    def run(self):
        self.loadData()
        model = Sequential()
        model.add(LSTM(256,return_sequences=True,input_shape=(8,256)))
        model.add(Dropout(0.25))
        model.add(LSTM(256,return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(256,return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(256,return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(256,return_sequences=True))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='tanh'))
        model.summary()
        model.compile(loss='mean_squared_error',optimizer='rmsprop', metrics=['accuracy'])
        
        history=model.fit(self.x_train, self.y_train, batch_size=256, epochs=1000, validation_data=(self.x_val, self.y_val))
        
        model.save("/Users/sean/Documents/Projects/Packing-Algorithm/model/absolute_lstm_num_3_layer_128_56_2_epochs_1000.h5")
        HistoryShow.showAccr(history)
        HistoryShow.showLoss(history)

    def loadData(self):
        # _input=pd.read_csv("/Users/sean/Documents/Projects/Data/input_seq.csv")
        # _output=pd.read_csv("/Users/sean/Documents/Projects/Data/output_relative_position.csv")
        # np_input=np.asarray([json.loads(_input["x_256"][i]) for i in range(0,5000)])
        # np_output=np.asarray([json.loads(_output["y"][i]) for i in range(0,5000)])

        file=pd.read_csv("/Users/sean/Documents/Projects/Data/8_lstm_test.csv")
        np_input=np.asarray([json.loads(file["x_256"][i]) for i in range(0,4700)])
        np_output=np.asarray([json.loads(file["y"][i]) for i in range(0,4700)])
        
        self.x_train=np_input[0:3700]
        self.y_train=np_output[0:3700]
        self.x_val=np_input[3700:4700]
        self.y_val=np_output[3700:4700]
    
    '''测试LSTM预测模型用 3.10'''
    def getPredictionRelative(self):
        model = load_model("/Users/sean/Documents/Projects/Packing-Algorithm/model/lstm_num_5_layer_128_56_2_epochs_70.h5")
        pre_train = pd.read_csv("/Users/sean/Documents/Projects/Data/pre_train.csv") # 读取形状
        _input = pd.read_csv("/Users/sean/Documents/Projects/Data/input_seq.csv") # 读取输入
        _output = pd.read_csv("/Users/sean/Documents/Projects/Data/output_relative_position.csv") # 读取输入
        
        
        # index=random.randint(4000,5000)
        index=4500

        polys=json.loads(pre_train["polys"][index]) # 形状
        X = np.array([json.loads(_input["x_256"][index])]) # 输入
        predicted_Y = model.predict(X, verbose=0)[0]*1500
        print(predicted_Y)
        Y=np.array(json.loads(_output["y"][index]))*1500
        print(Y)

        old_centroid=[0,0]
        for i,poly in enumerate(polys):
            # 获得初始的中心和预测的位置
            centroid_origin=GeoFunc.getPt(Polygon(poly).centroid)
            centroid_predicted=[Y[i][0]+old_centroid[0],Y[i][1]+old_centroid[1]] 

            # 获得新的形状并更新
            new_poly=GeoFunc.getSlide(poly,centroid_predicted[0]-centroid_origin[0],centroid_predicted[1]-centroid_origin[1])
            old_centroid=GeoFunc.getPt(Polygon(new_poly).centroid)

            PltFunc.addPolygon(poly)
            PltFunc.addPolygonColor(new_poly)

        PltFunc.showPlt()
    
    def getPredictionAbsolute(self):
        model = load_model("/Users/sean/Documents/Projects/Packing-Algorithm/model/absolute_lstm_num_8_layer_128_56_2_epochs_200.h5")
        
        file= pd.read_csv("/Users/sean/Documents/Projects/Data/8_lstm_test.csv") # 读取输入
        
        index=random.randint(3700,4700)
        index=3000

        polys=json.loads(file["polys"][index]) # 形状
        X = np.array([json.loads(file["x_256"][index])]) # 输入
        predicted_Y = model.predict(X, verbose=0)[0]*4000

        for i,poly in enumerate(polys):
            centroid_origin=GeoFunc.getPt(Polygon(poly).centroid)
            PltFunc.addPolygon(poly)

            new_poly=GeoFunc.getSlide(poly,predicted_Y[i][0]-centroid_origin[0],predicted_Y[i][1]-centroid_origin[1])
            PltFunc.addPolygonColor(new_poly)

        PltFunc.showPlt()



class HistoryShow(object):
    '''损失情况'''
    def showLoss(history):
        val_loss_values = history.history['val_loss']
        loss_values = history.history['loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'bo', color='#A6C8E0',label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', color='#A6C8E0',label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.clf()
    
    '''准确率'''
    def showAccr(history):
        acc = history.history['accuracy']
        epochs = range(1, len(acc) + 1)
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'bo', color='#A6C8E0',label='Training acc')
        plt.plot(epochs, val_acc, 'b', color='#A6C8E0',label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

if __name__=='__main__':
    lstm=LSTMPredict()
    lstm.run()
    # lstm.getPrediction()
    # lstm.getPredictionAbsolute()
