'''
网络训练情况
1. 标准化：形状全部转化为向量表示，收缩将1000作为收缩参数，后续获得其最大值，宽度为其
2. 宽度：暂时不考虑宽度输入，默认为1500，后续"宽度/10000"作为第一个输入参数
3. 输出：分别为重心的x/y坐标
4. 旋转角度：后续增加旋转角度的处理
5. 利用率：可以预测利用率的范围
6. 可行化：(a) 采用Minimize Overlap，形状全部向下平移 (b) 采用最低高度进行序列生成
'''
import numpy as np
import json
from keras.models import Sequential
from keras import models
from keras import layers
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,mapping
import random
import pandas as pd
import time
import csv
from polygon import GeoFunc,PltFunc
from sequence import BottomLeftFill
from feasible import getFeasibleByBottom

class trainModel(object):
   def __init__(self):
      pass
   
   def getTrainData(self):
      x_train,y_train,x_val,y_val=[],[],[],[]

      _input=pd.read_csv("/Users/sean/Documents/Projects/Data/input.csv")
      _output=pd.read_csv("/Users/sean/Documents/Projects/Data/output_position.csv")
      for i in range(0,4000):
         x_train.append(json.loads(_input["x_128"][i]))
         y_train.append(json.loads(_output["y_position"][i]))
      for i in range(4000,5001):
         x_val.append(json.loads(_input["x_128"][i]))
         y_val.append(json.loads(_output["y_position"][i]))

      self.trainModel(np.array(x_train),np.array(y_train),np.array(x_val),np.array(y_val))

   def testModel(self):
      '''
      测试训练结果, 使用未作为训练集的数据进行测试
      '''
      model = load_model("/Users/sean/Documents/Projects/Packing-Algorithm/model/new_128.h5")
      _input = pd.read_csv("/Users/sean/Documents/Projects/Data/input.csv")
      pre_train = pd.read_csv("/Users/sean/Documents/Projects/Data/pre_train.csv")

      index=random.randint(4000,5000)
      print(index)

      X = np.array([json.loads(_input["x_128"][index])])
      Y = model.predict(X, verbose=0)[0]

      # 获得预测结果
      polys=json.loads(pre_train["polys"][index])
      new_polys=[]
      for i,poly in enumerate(polys):
         centroid=GeoFunc.getPt(Polygon(poly).centroid)
         new_polys.append(GeoFunc.getSlide(poly,Y[i*2]*1500-centroid[0],Y[i*2+1]*3000-centroid[1]))
      
      # 获得可行解
      feasible_polys=getFeasibleByBottom(new_polys)

      # 获得最优解
      best_seq=[]
      order=json.loads(pre_train["seq"][index])
      for i in order:
         best_seq.append(polys[i])
      BottomLeftFill(1500,best_seq)

      for i in range(0,len(feasible_polys)):
         PltFunc.addPolygon(best_seq[i])
         PltFunc.addPolygonColor(feasible_polys[i])
      
      PltFunc.showPlt()
 
   def trainModel(self,x_train,y_train,x_val,y_val):
      '''
      训练模型
      '''
      print('Build ------------')
      model = Sequential()

      model.add(layers.Dense(640, activation='sigmoid', input_shape=(640,)))

      # model.add(layers.Dense(256, activation='sigmoid'))

      model.add(layers.Dense(128, activation='sigmoid'))

      model.add(layers.Dense(64, activation='sigmoid'))
      
      model.add(layers.Dense(10, activation='sigmoid'))

      # model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy']) 
      model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=['accuracy']) 

      # 使用fit()来训练网路
      print('Training ------------')
      history = model.fit(x_train, y_train, epochs=2000, batch_size=256,validation_data=(x_val, y_val))
      
      model.save("/Users/sean/Documents/Projects/Packing-Algorithm/model/new_256.h5")
      # model.save("/Users/sean/Documents/Projects/Packing-Algorithm/model/mse_epoch_250_sample_5000_rmsprop_128_64_batch_256.h5")

      print('Plot ------------')
      import matplotlib.pyplot as plt

      history_dict = history.history

      self.storeResult(history_dict)
      
      '''
         损失值判断
      '''
      val_loss_values = history_dict['val_loss']
      loss_values = history_dict['loss']
      epochs = range(1, len(loss_values) + 1)     
      plt.plot(epochs, loss_values, 'bo', color='#A6C8E0',label='Training loss')
      plt.plot(epochs, val_loss_values, 'b', color='#A6C8E0',label='Validation loss')
      plt.title('Training and validation loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.show()

      plt.clf()

      '''
         准确性
      '''
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
   
   def storeResult(self,history_dict):
      '''
      存储最终的训练结果
      '''
      with open("/Users/sean/Documents/Projects/Packing-Algorithm/model/record/"+time.asctime(time.localtime(time.time()))+"result.csv","a+") as csvfile: 
         writer = csv.writer(csvfile)
         for i in range(len(history_dict["loss"])):
            writer.writerows([[history_dict["accuracy"][i],history_dict["val_accuracy"][i],history_dict["loss"][i],history_dict["val_loss"][i]]])
   
   def getPt(self,point):
      mapping_result=mapping(point)
      return [mapping_result["coordinates"][0],mapping_result["coordinates"][1]]
   
if __name__ == '__main__':
   tm=trainModel()
   # tm.getTrainData()
   # tm.trainModel()
   # tm.modelRealTest()
   # tm.testModel()
   for i in range(100):
      tm.testModel()