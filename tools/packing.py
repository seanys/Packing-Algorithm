from tools.polygon import GeoFunc,NFP,Poly
from shapely.geometry import Polygon,Point,mapping,LineString
from shapely.ops import unary_union
from shapely import affinity
#from multiprocessing import Pool
import heuristic
import pyclipper 
import math
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import csv
import logging
import random
import copy
import os

def getNFP(poly1,poly2): # 这个函数必须放在class外面否则多进程报错
    nfp=NFP(poly1,poly2).nfp
    return nfp


class PackingUtil(object):
    
    @staticmethod
    def getInnerFitRectangle(poly,x,y):
        left_index,bottom_index,right_index,top_index=GeoFunc.checkBound(poly) # 获得边界
        new_poly=GeoFunc.getSlide(poly,-poly[left_index][0],-poly[bottom_index][1]) # 获得平移后的结果

        refer_pt=[new_poly[top_index][0],new_poly[top_index][1]]
        ifr_width=x-new_poly[right_index][0]
        ifr_height=y-new_poly[top_index][1]

        IFR=[refer_pt,[refer_pt[0]+ifr_width,refer_pt[1]],[refer_pt[0]+ifr_width,refer_pt[1]+ifr_height],[refer_pt[0],refer_pt[1]+ifr_height]]
        return IFR
    
class NFPAssistant(object):
    def __init__(self,polys,**kw):
        self.polys=PolyListProcessor.deleteRedundancy(copy.deepcopy(polys))
        self.area_list,self.first_vec_list,self.centroid_list=[],[],[] # 作为参考
        for poly in self.polys:
            P=Polygon(poly)
            self.centroid_list.append(GeoFunc.getPt(P.centroid))
            self.area_list.append(int(P.area))
            self.first_vec_list.append([poly[1][0]-poly[0][0],poly[1][1]-poly[0][1]])
        self.nfp_list=[[0]*len(self.polys) for i in range(len(self.polys))]
        self.load_history=False
        self.history_path=None
        self.history=None
        if 'history_path' in kw:
            self.history_path=kw['history_path']

        if 'load_history' in kw:
            if kw['load_history']==True:
                # 从内存中加载history 直接传递pandas的df对象 缩短I/O时间
                if 'history' in kw:
                    self.history=kw['history']
                self.load_history=True
                self.loadHistory()
        
        self.store_nfp=False
        if 'store_nfp' in kw:
            if kw['store_nfp']==True:
                self.store_nfp=True
        
        self.store_path=None
        if 'store_path' in kw:
            self.store_path=kw['store_path']

        if 'get_all_nfp' in kw:
            if kw['get_all_nfp']==True and self.load_history==False:
                self.getAllNFP()
        
        if 'fast' in kw: # 为BLF进行多进程优化
            if kw['fast']==True:
                self.res=[[0]*len(self.polys) for i in range(len(self.polys))]
                #pool=Pool()
                for i in range(1,len(self.polys)):
                    for j in range(0,i):
                        # 计算nfp(j,i)
                        #self.res[j][i]=pool.apply_async(getNFP,args=(self.polys[j],self.polys[i]))
                        self.nfp_list[j][i]=GeoFunc.getSlide(getNFP(self.polys[j],self.polys[i]),-self.centroid_list[j][0],-self.centroid_list[j][1])
                # pool.close()
                # pool.join()
                # for i in range(1,len(self.polys)):
                #     for j in range(0,i):
                #         self.nfp_list[j][i]=GeoFunc.getSlide(self.res[j][i].get(),-self.centroid_list[j][0],-self.centroid_list[j][1])

    def loadHistory(self):
        if not self.history:
            if not self.history_path:
                path="/Users/sean/Documents/Projects/Packing-Algorithm/record/nfp.csv"
            else:
                path=self.history_path
            df = pd.read_csv(path,header=None)
        else:
            df = self.history
        for index in range(df.shape[0]):
            i=self.getPolyIndex(json.loads(df[0][index]))
            j=self.getPolyIndex(json.loads(df[1][index]))
            if i>=0 and j>=0:
                self.nfp_list[i][j]=json.loads(df[2][index])
        # print(self.nfp_list)
        
    # 获得一个形状的index
    def getPolyIndex(self,target):
        area=int(Polygon(target).area)
        first_vec=[target[1][0]-target[0][0],target[1][1]-target[0][1]]
        area_index=PolyListProcessor.getIndexMulti(area,self.area_list)
        if len(area_index)==1: # 只有一个的情况
            return area_index[0]
        else:
            vec_index=PolyListProcessor.getIndexMulti(first_vec,self.first_vec_list)
            index=[x for x in area_index if x in vec_index]
            if len(index)==0:
                return -1
            return index[0] # 一般情况就只有一个了
    
    # 获得所有的形状
    def getAllNFP(self):
        nfp_multi=False 
        if nfp_multi==True:
            tasks=[(main,adjoin) for main in self.polys for adjoin in self.polys]
            res=pool.starmap(NFP,tasks)
            for k,item in enumerate(res):
                i=k//len(self.polys)
                j=k%len(self.polys)
                self.nfp_list[i][j]=GeoFunc.getSlide(item.nfp,-self.centroid_list[i][0],-self.centroid_list[i][1])
        else:
            for i,poly1 in enumerate(self.polys):
                for j,poly2 in enumerate(self.polys):
                    nfp=NFP(poly1,poly2).nfp
                    #NFP(poly1,poly2).showResult()
                    self.nfp_list[i][j]=GeoFunc.getSlide(nfp,-self.centroid_list[i][0],-self.centroid_list[i][1])
        if self.store_nfp==True:
            self.storeNFP()
    
    def storeNFP(self):
        if self.store_path==None:
            path="/Users/sean/Documents/Projects/Packing-Algorithm/record/nfp.csv"
        else:
            path=self.store_path
        with open(path,"a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(self.polys)):
                for j in range(len(self.polys)):
                    writer.writerows([[self.polys[i],self.polys[j],self.nfp_list[i][j]]])

    # 输入形状获得NFP
    def getDirectNFP(self,poly1,poly2,**kw):
        if 'index' in kw:
            i=kw['index'][0]
            j=kw['index'][1]
            centroid=GeoFunc.getPt(Polygon(self.polys[i]).centroid)
        else:
            # 首先获得poly1和poly2的ID
            i=self.getPolyIndex(poly1)
            j=self.getPolyIndex(poly2)
            centroid=GeoFunc.getPt(Polygon(poly1).centroid)
        # 判断是否计算过并计算nfp
        if self.nfp_list[i][j]==0:
            nfp=NFP(poly1,poly2).nfp
            #self.nfp_list[i][j]=GeoFunc.getSlide(nfp,-centroid[0],-centroid[1])
            if self.store_nfp==True:
                with open("/Users/sean/Documents/Projects/Packing-Algorithm/record/nfp.csv","a+") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[poly1,poly2,nfp]])
            return nfp
        else:
            return GeoFunc.getSlide(self.nfp_list[i][j],centroid[0],centroid[1])

class PolyListProcessor(object):
    @staticmethod
    def getPolyObjectList(polys,allowed_rotation):
        '''
        将Polys和允许旋转的角度转化为poly_lists
        '''
        poly_list=[]
        for i,poly in enumerate(polys):
            poly_list.append(Poly(i,poly,allowed_rotation))
        return poly_list

    @staticmethod
    def getPolysVertices(_list):
        '''排序结束后会影响'''
        polys=[]
        for i in range(len(_list)):
            polys.append(_list[i].poly)
        return polys
    
    @staticmethod
    def getPolysVerticesCopy(_list):
        '''不影响list内的形状'''
        polys=[]
        for i in range(len(_list)):
            polys.append(copy.deepcopy(_list[i].poly))
        return polys

    @staticmethod
    def getPolyListIndex(poly_list):
        index_list=[]
        for i in range(len(poly_list)):
            index_list.append(poly_list[i].num)
        return index_list
    
    @staticmethod
    def getIndex(item,_list):
        for i in range(len(_list)):
            if item==_list[i]:
                return i
        return -1
    
    @staticmethod
    def getIndexMulti(item,_list):
        index_list=[]
        for i in range(len(_list)):
            if item==_list[i]:
                index_list.append(i)
        return index_list

    @staticmethod
    def packingLength(poly_list,history_index_list,history_length_list,width,**kw):
        polys=PolyListProcessor.getPolysVertices(poly_list)
        index_list=PolyListProcessor.getPolyListIndex(poly_list)
        length=0
        check_index=PolyListProcessor.getIndex(index_list,history_index_list)
        if check_index>=0:
            length=history_length_list[check_index]
        else:
            try:
                if 'NFPAssistant' in kw:
                    length=heuristic.BottomLeftFill(width,polys,NFPAssistant=kw['NFPAssistant']).contain_length
                else:
                    length=heuristic.BottomLeftFill(width,polys).contain_length
            except:
                print('出现Self-intersection')
                length=99999
            history_index_list.append(index_list)
            history_length_list.append(length)
        return length

    @staticmethod
    def randomSwap(poly_list,target_id):
        new_poly_list=copy.deepcopy(poly_list)

        swap_with = int(random.random() * len(new_poly_list))
        
        item1 = new_poly_list[target_id]
        item2 = new_poly_list[swap_with]
            
        new_poly_list[target_id] = item2
        new_poly_list[swap_with] = item1
        return new_poly_list

    @staticmethod
    def randomRotate(poly_list,min_angle,target_id):
        new_poly_list=copy.deepcopy(poly_list)

        index = random.randint(0,len(new_poly_list)-1)
        heuristic.RatotionPoly(min_angle).rotation(new_poly_list[index].poly)
        return new_poly_list

    @staticmethod
    def showPolyList(width,poly_list):
        blf=heuristic.BottomLeftFill(width,PolyListProcessor.getPolysVertices(poly_list))
        blf.showAll()

    @staticmethod
    def deleteRedundancy(_arr):
        new_arr = []
        for item in _arr:
            if not item in new_arr:
                new_arr.append(item)
        return new_arr

    @staticmethod
    def getPolysByIndex(index_list,poly_list):
        choosed_poly_list=[]
        for i in index_list:
            choosed_poly_list.append(poly_list[i])
        return choosed_poly_list
