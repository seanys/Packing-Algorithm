import pandas as pd # 读csv
import csv # 写csv
import numpy as np # 数据处理
import os
import re
import json
import time
import random
import math
from tools.polygon import PltFunc,GeoFunc,NFP,getConvex,getData
from shapely.geometry import Polygon,mapping
from tools.vectorization import vectorFunc
from sequence import BottomLeftFill,NFPAssistant
import itertools

def getTrainData(point_num):
    """
    point_num:取样点个数
    return:x,y,x_test,y_test
    """
    if os.getlogin()=='Prinway':
        path=r'D:\\Tongji\\Nesting\\Data\\'
    else:
        path='/Users/sean/Documents/Projects/Data/'
    X,Y,x_test,y_test=[],[],[],[]
    _input=pd.read_csv(path+'input_seq.csv')
    _output=pd.read_csv(path+'output_order.csv')
    # 训练集
    for i in range(0,4000):
        X.append(json.loads(_input["x_"+str(point_num)][i]))
        Y.append(json.loads(_output["y_order"][i]))
    # 测试集
    for i in range(4000,5000):
        x_test.append(json.loads(_input["x_"+str(point_num)][i]))
        y_test.append(json.loads(_output["y_order"][i]))

    return np.array(X),np.array(Y),np.array(x_test),np.array(y_test)


class GetBestSeq(object):
    def __init__(self,width,polys):
        self.polys=polys
        self.width=width
        self.NFPAssistant=NFPAssistant(polys)
    
    # 获得面积降序排列的形状结果
    def getDrease(self):
        poly_list=[]
        for poly in polys:
            poly_list.append([poly,Polygon(poly).area])
        poly_list=sorted(poly_list, key = operator.itemgetter(1), reverse = True) # 排序，包含index
        dec_polys=[]
        for item in poly_list:
            dec_polys.append(item[0])
        blf=BottomLeftFill(width,dec_polys)
        return blf.polygons
    
    # 从所有的排列中选择出最合适的
    def chooseFromAll(self):
        all_com=list(itertools.permutations([(i) for i in range(len(self.polys))]))
        min_height=999999999
        best_order=[]
        for item in all_com:
            seq=self.getPolys(item)
            height=BottomLeftFill(self.width,seq,NFPAssistant=self.NFPAssistant).contain_height
            if height<min_height:
                best_order=item
                min_height=height
        area=0
        for poly in self.polys:
            area=area+Polygon(poly).area
        use_ratio=area/(self.width*min_height)
        return best_order,min_height,use_ratio
    
    def getPolys(self,seq):
        seq_polys=[]
        for i in seq:
            seq_polys.append(self.polys[i])
        return seq_polys

class getShape(object):
    def getConvexRandom():
        polygon=[]
        num=10
        for i in range(0,num):
            # radian=(2/num)*math.pi*i+math.pi*random.randrange(0,5,1)/12 # convex 4的角度
            radian=(2/num)*math.pi*i+math.pi*random.randrange(0,3,1)/(num*2) # convex num>4的角度
            radius=random.randrange(200,500,100)
            pt=[radius*math.cos(radian),radius*math.sin(radian)]
            polygon.append(pt)
        geoFunc.slidePoly(polygon,750,750)
        storePolygon(polygon,num=num)
        pltFunc.addPolygon(polygon)
        pltFunc.showPlt()

# Preprocess train data
class trainDataProcess(object):
    def __init__(self):
        self.getBLFTrain()
    
    def addBestSeq(self):
        pre_train = pd.read_csv("/Users/sean/Documents/Projects/Data/pre_train.csv")
        with open("/Users/sean/Documents/Projects/Data/new_pre_train.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for row_num in range(0,5001):
                print(row_num)
                polys=json.loads(pre_train["polys"][row_num])
                best_seq=[]
                order=json.loads(pre_train["seq"][row_num])
                for i in order:
                    best_seq.append(polys[i])
                result=BottomLeftFill(1500,best_seq).polygons
                writer.writerows([[pre_train["time"][row_num],json.loads(pre_train["poly_indexs"][row_num]),pre_train["width"][row_num],pre_train["num"][row_num],pre_train["height"][row_num],pre_train["use_ratio"][row_num],json.loads(pre_train["seq"][row_num]),polys,result]])
    
    # 输出相对位置，测试Pointer Network
    def getOutputOrder(self):
        pre_train = pd.read_csv("/Users/sean/Documents/Projects/Data/pre_train.csv")
        with open("/Users/sean/Documents/Projects/Data/output_order.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(0,5001):
                order=json.loads(pre_train["seq"][i])
                writer.writerows([[i,order]])
    
    # 输出位置，测试
    def getOutputPosition(self):
        pre_train = pd.read_csv("/Users/sean/Documents/Projects/Data/pre_train.csv")
        with open("/Users/sean/Documents/Projects/Data/output_position.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(0,5001):
                polys=json.loads(pre_train["best_result"][i])
                position_y=[]
                for poly in polys:
                    centroid=GeoFunc.getPt(Polygon(poly).centroid)
                    position_y=position_y+[centroid[0]/1500,centroid[1]/3000]
                writer.writerows([[i,position_y]])
                
    # 相对位置
    def getOutputRelativePosition(self):
        # 读取预处理数据
        pre_train = pd.read_csv("/Users/sean/Documents/Projects/Data/pre_train.csv") 
        # 获得相对位置并存储
        with open("/Users/sean/Documents/Projects/Data/output_relative_position.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(0,5000):
                polys=json.loads(pre_train["polys"][i])
                centroid=GeoFunc.getPt(Polygon(polys[0]).centroid)
                relative_position=[[centroid[0]/1500,centroid[1]/1500]]
                for j in range(1,len(polys)):
                    centroid0=GeoFunc.getPt(Polygon(polys[j-1]).centroid)
                    centroid1=GeoFunc.getPt(Polygon(polys[j]).centroid)
                    relative_position.append([(centroid1[0]-centroid0[0])/1500,(centroid1[1]-centroid0[1])/1500])
                writer.writerows([[i,relative_position]])

    # 获得输入的形状
    def getInput(self):
        pre_train = pd.read_csv("/Users/sean/Documents/Projects/Data/pre_train.csv")
        convex = pd.read_csv("/Users/sean/Documents/Projects/Data/convex_vec.csv")
        with open("/Users/sean/Documents/Projects/Data/input.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            labels=["vec_8","vec_16","vec_32","vec_64","vec_128","vec_256"]
            for i in range(5000,5001):
                poly_indexs=json.loads(pre_train["poly_indexs"][i])
                all_x=[]
                for target_lable in labels:
                    x=[]
                    for index in poly_indexs:
                        x=x+self.normVec(json.loads(convex[target_lable][index],1200))
                    all_x.append(x)
                writer.writerows([[i,all_x[0],all_x[1],all_x[2],all_x[3],all_x[4],all_x[5]]])
    
    # 获得输入的形状
    def getInputSeq(self):
        pre_train = pd.read_csv("/Users/sean/Documents/Projects/Data/pre_train.csv")
        convex = pd.read_csv("/Users/sean/Documents/Projects/Data/convex_vec.csv")
        with open("/Users/sean/Documents/Projects/Data/input_seq.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            labels=["vec_8","vec_16","vec_32","vec_64","vec_128","vec_256"]
            for i in range(1000,5001):
                poly_indexs=json.loads(pre_train["poly_indexs"][i])
                all_x=[]
                for target_lable in labels:
                    x=[]
                    for index in poly_indexs:
                        x.append(self.normVec(json.loads(convex[target_lable][index]),500))
                    all_x.append(x)
                writer.writerows([[i,all_x[0],all_x[1],all_x[2],all_x[3],all_x[4],all_x[5]]])

    # 向量标准化
    def normVec(self,_arr,divisor):
        new_arr=[]
        for item in _arr:
            if item==9999999:
                new_arr.append(0)
            else:
                new_arr.append(item/divisor) # 全部正数
        return new_arr

    # 位置标准化    
    def normPosi(self,_arr):
        new_arr=[]
        for item in _arr:
            new_arr.append(item/2400)
        return new_arr
    
    def getBLFTrain(self):
        width=2000
        num=8
        convex = pd.read_csv("/Users/sean/Documents/Projects/Data/convex_vec.csv")
        labels=["vec_64","vec_128","vec_256"]
        for i in range(0,10000):
            poly_index,polys=getConvex(num=num,with_index=True)
            blf=BottomLeftFill(width,polys)
            new_polys=blf.polygons
            y=[]
            for poly in new_polys:
                centroid=GeoFunc.getCentroid(poly)
                y.append(self.normVec(centroid,4000))
            all_x=[]
            for label in labels:
                x=[]
                for index in poly_index:
                    x.append(self.normVec(json.loads(convex[label][index]),1000))
                all_x.append(x)
            with open("/Users/sean/Documents/Projects/Data/8_lstm_test.csv","a+") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([[time.asctime(time.localtime(time.time())),poly_index,new_polys,y,all_x[0],all_x[1],all_x[2]]])
        

    def getAllData(self):
        width=1500
        num=5
        # 获得足够的形状组合
        for i in range(0,10000):
            poly_index,polys=getConvex(num=num,with_index=True)
            best_order,min_height,use_ratio=GetBestSeq(width,polys).chooseFromAll()
            with open("/Users/sean/Documents/Projects/Data/pre_train.csv","a+") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([[time.asctime(time.localtime(time.time())),poly_index,width,num,min_height,use_ratio,[i for i in best_order],polys]])


if __name__ == '__main__':
    trainDataProcess()
