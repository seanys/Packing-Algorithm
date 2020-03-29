"""
该文件实现了主要基于序列的排样算法
-----------------------------------
Created on Wed Dec 11, 2019
@author: seanys,prinway
-----------------------------------
"""
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from tools.polygon import GeoFunc,NFP,PltFunc,RatotionPoly,getData,getConvex,Poly
from tools.packing import PackingUtil,NFPAssistant,PolyListProcessor
import json
from shapely.geometry import Polygon,mapping
from shapely import affinity
import csv
import time
import multiprocessing
import datetime
import random
import copy


cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
nfp_multi=True
ga_multi=False

class BottomLeftFill(object):
    def __init__(self,width,original_polygons,**kw):
        self.choose_nfp=False
        self.width=width
        self.length=150000 # 代表长度
        self.polygons=copy.deepcopy(original_polygons)
        self.placeFirstPoly()
        if 'NFPAssistant' in kw:
            self.NFPAssistant=kw["NFPAssistant"]
        else:
            self.NFPAssistant=None
        if 'vertical' in kw:
            self.vertical=True
        else:
            self.vertical=False

        # for i in range(1,3):
        for i in range(1,len(self.polygons)):
            # print("##############################放置第",i+1,"个形状#################################")
            self.placePoly(i)
        
        self.getLength()

    def placeFirstPoly(self):
        poly=self.polygons[0]
        left_index,bottom_index,right_index,top_index=GeoFunc.checkBound(poly) # 获得边界        
        GeoFunc.slidePoly(poly,-poly[left_index][0],-poly[bottom_index][1]) # 平移到左下角

    def placePoly(self,index):
        adjoin=self.polygons[index]
        # 是否垂直
        if self.vertical==True:
            ifr=PackingUtil.getInnerFitRectangle(self.polygons[index],self.width,self.length)
        else:
            ifr=PackingUtil.getInnerFitRectangle(self.polygons[index],self.length,self.width)            
        differ_region=Polygon(ifr)
        
        for main_index in range(0,index):
            main=self.polygons[main_index]
            if self.NFPAssistant==None:
                nfp=NFP(main,adjoin).nfp
            else:
                nfp=self.NFPAssistant.getDirectNFP(main,adjoin)
            differ_region=differ_region.difference(Polygon(nfp))

        differ=GeoFunc.polyToArr(differ_region)

        differ_index=self.getBottomLeft(differ)
        refer_pt_index=GeoFunc.checkTop(adjoin)
        GeoFunc.slideToPoint(self.polygons[index],adjoin[refer_pt_index],differ[differ_index])        

    def getBottomLeft(self,poly):
        '''
        获得左底部点，优先左侧，有多个左侧选择下方
        '''
        bl=[] # bottom left的全部点
        _min=999999
        # 选择最左侧的点
        for i,pt in enumerate(poly):
            pt_object={
                    "index":i,
                    "x":pt[0],
                    "y":pt[1]
            }
            if self.vertical==True:
                target=pt[1]
            else:
                target=pt[0]
            if target<_min:
                _min=target
                bl=[pt_object]
            elif target==_min:
                bl.append(pt_object)
        if len(bl)==1:
            return bl[0]["index"]
        else:
            if self.vertical==True:
                target="x"                
            else:
                target="y"
            _min=bl[0][target]
            one_pt=bl[0]
            for pt_index in range(1,len(bl)):
                if bl[pt_index][target]<_min:
                    one_pt=bl[pt_index]
                    _min=one_pt["y"]
            return one_pt["index"]

    def showAll(self):
        # for i in range(0,2):
        for i in range(0,len(self.polygons)):
            PltFunc.addPolygon(self.polygons[i])
        length=max(self.width,self.contain_length)
        # PltFunc.addLine([[self.width,0],[self.width,self.contain_height]],color="blue")
        PltFunc.showPlt(width=max(length,self.width),height=max(length,self.width))    

    def getLength(self):
        _max=0
        for i in range(0,len(self.polygons)):
            if self.vertical==True:
                extreme_index=GeoFunc.checkTop(self.polygons[i])
            else:
                extreme_index=GeoFunc.checkRight(self.polygons[i])
            extreme=self.polygons[i][extreme_index][1]
            if extreme>_max:
                _max=extreme
        self.contain_length=_max
        # PltFunc.addLine([[0,self.contain_length],[self.width,self.contain_length]],color="blue")
        return _max

class TOPOS(object):
    '''
    TOPOS启发式算法：将形状一个个放入，动态移动整体的位置
    '''
    def __init__(self,original_polys,width):
        self.polys=original_polys
        self.cur_polys=[]
        self.width=width
        self.NFPAssistant=NFPAssistant(self.polys,store_nfp=False,get_all_nfp=True,load_history=True)
        
        self.run()

    def run(self):
        self.cur_polys.append(GeoFunc.getSlide(self.polys,1000,1000))
        for i in range(1,len(self.polys)):
            feasible_border=Polygon(self.cur_polys[0])

            # 一个个计算重叠区域
            for fixed_poly in self.cur_polys:
                nfp=self.NFPAssistant.getDirectNFP(fixed_poly,self.polys[i])
                feasible_border=feasible_border.union(Polygon(nfp))
            
            # 将最终计算结果所有的顶点转化为向量
            border=GeoFunc.polyToArr(feasible_border)
        
        
    def show(self):
        for poly in self.cur_polys:
            PltFunc.addPolygon(poly)
        PltFunc.showPolys()

    
if __name__=='__main__':
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    nfp_multi=True
    ga_multi=False
    starttime = datetime.datetime.now()
    # polys=getConvex(num=5)
    polys=getData()
    # poly_list=PolyListProcessor.getPolyObjectList(polys,[0])
    TOPOS(polys,1500)

    # 计算NFP时间
    # print(datetime.datetime.now(),"开始计算NFP")
    # nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    # print(datetime.datetime.now(),"计算完成NFP")
    # bfl=BottomLeftFill(1500,polys,vertical=True,NFPAssistant=nfp_ass)
    
    # print(datetime.datetime.now(),"计算完成BLF")

    # GA(poly_list)
    # SA(poly_list)

    # GetBestSeq(1000,getConvex(num=5),"decrease")
    endtime = datetime.datetime.now()
    print (endtime - starttime)
    # bfl.showAll()