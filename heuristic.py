"""
该文件实现了主要基于序列的排样算法
-----------------------------------
Created on Wed Dec 11, 2019
@author: seanys,prinway
-----------------------------------
"""
from shapely.geometry import Polygon,mapping
from shapely import affinity
from tools.geo_assistant import GeometryAssistant
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from tools.polygon import GeoFunc,NFP,PltFunc,RatotionPoly,getData,getConvex,Poly
import tools.packing as packing
import json
import csv
import time
import multiprocessing
import datetime
import random
import copy

class BottomLeftFill(object):
    def __init__(self,width,original_polygons,**kw):
        self.choose_nfp=False
        self.width=width
        self.length=150000 # 代表长度
        self.contain_length=3000
        self.polygons=original_polygons
        self.placeFirstPoly()
        self.NFPAssistant=None

        if 'NFPAssistant' in kw:
            self.NFPAssistant=kw["NFPAssistant"]
        # else:
        #     # 若未指定外部NFPasst则内部使用NFPasst开多进程
        #     self.NFPAssistant=packing.NFPAssistant(self.polygons,fast=True)
        self.vertical=False
        if 'vertical' in kw:
                self.vertical=kw['vertical']
        if 'rectangle' in kw:
            self.rectangle=True
        else:
            self.rectangle=False
        # for i in range(1,3):
        for i in range(1,len(self.polygons)):
            print("##############################放置第",i+1,"个形状#################################")
            self.placePoly(i)
            # self.showAll()

        self.getLength()
        print(self.polygons)
        # self.showAll()

    def placeFirstPoly(self):
        poly = self.polygons[0]
        left_index,bottom_index,right_index,top_index = GeoFunc.checkBound(poly) # 获得边界        
        GeoFunc.slidePoly(poly,-poly[left_index][0],-poly[bottom_index][1]) # 平移到左下角

    def placePoly(self,index):
        adjoin = self.polygons[index]
        # 是否垂直
        if self.vertical == True:
            ifr = packing.PackingUtil.getInnerFitRectangle(self.polygons[index],self.width,self.length)
        else:
            ifr = packing.PackingUtil.getInnerFitRectangle(self.polygons[index],self.length,self.width)            
        differ_region = Polygon(ifr)
        
        for main_index in range(0,index):
            main = self.polygons[main_index]
            if self.NFPAssistant == None:
                nfp = NFP(main,adjoin,rectangle = self.rectangle).nfp
            else:
                nfp = self.NFPAssistant.getDirectNFP(main,adjoin)
            nfp_poly = Polygon(nfp)
            try:
                differ_region = differ_region.difference(nfp_poly)
            except:
                print(differ_region)
                print(nfp_poly)
                print('NFP failure, polys and nfp are:')
                print([main,adjoin])
                print(nfp)
                self.showAll()
                self.showPolys([main]+[adjoin]+[nfp])
                # print('NFP loaded from: ',self.NFPAssistant.history_path)

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
        length = max(self.width,self.contain_length)
        # PltFunc.addLine([[self.width,0],[self.width,self.contain_height]],color="blue")
        PltFunc.showPlt(width=max(length,self.width),height=max(length,self.width),minus=100)    

    def showPolys(self,polys):
        for i in range(0,len(polys)-1):
            PltFunc.addPolygon(polys[i])
        PltFunc.addPolygonColor(polys[len(polys)-1])
        length = max(self.width,self.contain_length)
        PltFunc.showPlt(width=max(length,self.width),height=max(length,self.width),minus=200)    

    def getLength(self):
        _max=0
        for i in range(0,len(self.polygons)):
            if self.vertical==True:
                extreme_index=GeoFunc.checkTop(self.polygons[i])
                extreme=self.polygons[i][extreme_index][1]
            else:
                extreme_index=GeoFunc.checkRight(self.polygons[i])
                extreme=self.polygons[i][extreme_index][0]
            if extreme>_max:
                _max=extreme
        self.contain_length=_max
        # PltFunc.addLine([[0,self.contain_length],[self.width,self.contain_length]],color="blue")
        return _max

class TOPOS(object):
    '''
    TOPOS启发式算法：将形状一个个放入，动态移动整体的位置，该算法参考Bennell的TOPOS Revised
    问题：中线情况、好像有一些bug
    '''
    def __init__(self,original_polys,width):
        self.polys=original_polys
        self.cur_polys=[]
        self.width=width
        self.NFPAssistant=packing.NFPAssistant(self.polys,store_nfp=False,get_all_nfp=True,load_history=True)
        
        self.run()

    def run(self):
        self.cur_polys.append(GeoFunc.getSlide(self.polys[0],1000,1000)) # 加入第一个形状
        self.border_left,self.border_right,self.border_bottom,self.border_top=0,0,0,0 # 初始化包络长方形
        self.border_height,self.border_width=0,0
        for i in range(1,len(self.polys)):
            # 更新所有的边界情况
            self.updateBound()

            # 计算NFP的合并情况
            feasible_border=Polygon(self.cur_polys[0])
            for fixed_poly in self.cur_polys:
                nfp=self.NFPAssistant.getDirectNFP(fixed_poly,self.polys[i])
                feasible_border=feasible_border.union(Polygon(nfp))
            
            # 获得所有可行的点
            feasible_point=self.chooseFeasiblePoint(feasible_border)
            
            # 获得形状的左右侧宽度
            poly_left_pt,poly_bottom_pt,poly_right_pt,poly_top_pt=GeoFunc.checkBoundPt(self.polys[i])
            poly_left_width,poly_right_width=poly_top_pt[0]-poly_left_pt[0],poly_right_pt[0]-poly_top_pt[0]

            # 逐一遍历NFP上的点，选择可行且宽度变化最小的位置
            min_change=999999999999
            target_position=[]
            for pt in feasible_point:
                change=min_change
                if pt[0]-poly_left_width>=self.border_left and pt[0]+poly_right_width<=self.border_right:
                    # 形状没有超出边界，此时min_change为负
                    change=min(self.border_left-pt[0],self.border_left-pt[0])
                elif min_change>0:
                    # 形状超出了左侧或右侧边界，若变化大于0，则需要选择左右侧变化更大的值
                    change=max(self.border_left-pt[0]+poly_left_width,pt[0]+poly_right_width-self.border_right)
                else:
                    # 有超出且min_change<=0的时候不需要改变
                    pass

                if change<min_change:
                    min_change=change
                    target_position=pt
            
            # 平移到最终的位置
            reference_point=self.polys[i][GeoFunc.checkTop(self.polys[i])]
            self.cur_polys.append(GeoFunc.getSlide(self.polys[i],target_position[0]-reference_point[0],target_position[1]-reference_point[1]))

        self.slideToBottomLeft()
        self.showResult()

    
    def updateBound(self):
        '''
        更新包络长方形
        '''
        border_left,border_bottom,border_right,border_top=GeoFunc.checkBoundValue(self.cur_polys[-1])
        if border_left<self.border_left:
            self.border_left=border_left
        if border_bottom<self.border_bottom:
            self.border_bottom=border_bottom
        if border_right>self.border_right:
            self.border_right=border_right
        if border_top>self.border_top:
            self.border_top=border_top
        self.border_height=self.border_top-self.border_bottom
        self.border_width=self.border_right-self.border_left
    
    def chooseFeasiblePoint(self,border):
        '''选择可行的点'''
        res=mapping(border)
        _arr=[]
        if res["type"]=="MultiPolygon":
            for poly in res["coordinates"]:
                _arr=_arr+self.feasiblePoints(poly)
        else:
            _arr=_arr+self.feasiblePoints(res["coordinates"][0])
        
        return _arr
    
    def feasiblePoints(self,poly):
        '''
        1. 将Polygon对象转化为点
        2. 超出Width范围的点排除
        3. 直线与边界的交点选入
        '''
        result=[]
        for pt in poly:
            # (1) 超出了上侧&总宽度没有超过
            feasible1=pt[1]-self.border_top>0 and pt[1]-self.border_top+self.border_height<=self.width
            # (2) 超过了下侧&总宽度没有超过
            feasible2=self.border_bottom-pt[1]>0 and self.border_bottom-pt[1]+self.border_heigt<=self.width
            # (3) Top和bottom的内部
            feasible3=pt[1]<=self.border_top and pt[1]>=self.border_bottom
            if feasible1==True or feasible2==True or feasible3==True:
                result.append([pt[0],pt[1]])
        return result

    def slideToBottomLeft(self):
        '''移到最左下角位置'''
        for poly in self.cur_polys:
            GeoFunc.slidePoly(poly,-self.border_left,-self.border_bottom)

    def showResult(self):
        '''显示排样结果'''
        for poly in self.cur_polys:
            PltFunc.addPolygon(poly)
        PltFunc.showPlt(width=2000,height=2000)

class newNFPAssistant(object):
    '''
    处理排样具体数据集加入，输入是形状，不考虑旋转！！
    '''
    def __init__(self,set_name,allowed_rotation):
        self.set_name = set_name
        self.allowed_rotation = allowed_rotation
        self.all_polys = pd.read_csv("data/" + self.set_name + "_orientation.csv") 
        self.all_nfps = pd.read_csv("data/" + self.set_name + "_nfp.csv")

    def getDirectNFP(self,main,adjoin):
        # 分别为固定i/移动j的形状
        i,j = self.judgeType(main),self.judgeType(adjoin)
        row = self.all_polys.shape[0]*self.allowed_rotation*self.allowed_rotation*i + self.allowed_rotation*self.allowed_rotation*j + self.allowed_rotation*0 + 1*0
        bottom_pt = GeometryAssistant.getBottomPoint(main)
        nfp = GeometryAssistant.getSlide(json.loads(self.all_nfps["nfp"][row]), bottom_pt[0], bottom_pt[1])
        return nfp 

    def judgeType(self,poly):
        # 判断形状类别，用于计算具体的NFP
        area = int(Polygon(poly).area)
        # print(area)
        for i in range(self.all_polys.shape[0]):
            new_poly = json.loads(self.all_polys["o_0"][i])
            test_poly_area = Polygon(new_poly).area
            if abs(test_poly_area - area) < 2 and abs((new_poly[1][0] - new_poly[0][0]) - (poly[1][0] - poly[0][0])) < 2:
                return i
        print("NFP错误")
    
index = 11
targets = [{
        "index" : 0,
        "name" : "blaz",
        "scale" : 10,
        "allowed_rotation": 2,
        "width": 750
    },{
        "index" : 1,
        "name" : "shapes2_clus",
        "scale" : 1,
        "allowed_rotation": 2,
        "width": 750
    },{
        "index" : 2,
        "name" : "shapes0",
        "scale" : 20,
        "allowed_rotation": 1,
        "width": 800
    },{
        "index" : 3,
        "name" : "marques",
        "scale" : 10,
        "allowed_rotation": 2,
        "width": 1040
    },{
        "index" : 4,
        "name" : "mao",
        "scale" : 1,
        "allowed_rotation": 4,
        "width": 2550
    },{
        "index" : 5,
        "name" : "shirts",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 800
    },{
        "index" : 6,
        "name" : "albano",
        "scale" : 0.2,
        "allowed_rotation": 2,
        "width": 980
    },{
        "index" : 7,
        "name" : "shapes1",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 800
    },{
        "index" : 8,
        "name" : "dagli_clus",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 1200
    },{
        "index" : 9,
        "name" : "jakobs1_clus",
        "scale" : 20,
        "allowed_rotation": 4,
        "width": 800
    },{
        "index" : 10,
        "name" : "trousers",
        "scale" : 10,
        "allowed_rotation": 2,
        "width": 790
    },{
        "index" : 11,
        "name" : "jakobs2_clus",
        "scale" : 10,
        "allowed_rotation": 4,
        "width": 700
    }]

def getDataNew():
    print(targets[index]["name"])
    print("缩放",targets[index]["scale"],"倍")

    df = pd.read_csv("data/" + targets[index]["name"] + ".csv")
    polygons=[]
    polys_type = []
    for i in range(0,df.shape[0]):
        for j in range(0,df['num'][i]):
            polys_type.append(i)
            poly = json.loads(df['polygon'][i])
            GeoFunc.normData(poly,targets[index]["scale"])
            polygons.append(poly)
    print(polys_type)
    return polygons

if __name__=='__main__':
    polys = getDataNew()
    
    total_area = 0
    for poly in polys:
        total_area = total_area + Polygon(poly).area
    print("total_area:",total_area)
    print("number:",len(polys))
    print([0 for i in range(len(polys))])

    # 计算NFP时间
    # nfp_ass = packing.NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)

    nfp_ass = newNFPAssistant(targets[index]["name"],allowed_rotation=targets[index]["allowed_rotation"])
    
    # nfp_ass.getDirectNFP(polys[10],polys[12])

    bfl = BottomLeftFill(targets[index]["width"],polys,vertical=False,NFPAssistant=nfp_ass)

    bfl.showAll()