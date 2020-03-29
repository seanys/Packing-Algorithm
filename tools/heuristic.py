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
    TOPOS启发式算法：将形状一个个放入，动态移动整体的位置，该算法参考Bennell的TOPOS Revised
    待办：中间位置的情况
    '''
    def __init__(self,original_polys,width):
        self.polys=original_polys
        self.cur_polys=[]
        self.width=width
        self.NFPAssistant=NFPAssistant(self.polys,store_nfp=False,get_all_nfp=True,load_history=True)
        
        self.run()

    def run(self):
        self.cur_polys.append(GeoFunc.getSlide(self.polys[0],1000,1000)) # 加入第一个形状
        self.border_left,self.border_right,self.border_bottom,self.border_top=0,0,0,0 # 初始化包络长方形
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
            poly_left_pt,poly_bottom_pt,poly_right_pt,poly_top_pt=GeoFunc.checkBoundArray(self.polys[i])
            poly_left_width,poly_right_width=poly_top_pt[0]-poly_left_pt[0],poly_right_pt[0]-poly_top_pt[0]

            # 逐一遍历NFP上的点，选择可行且宽度变化最小的位置
            min_change=999999999999
            target_position=[]
            for pt in feasible_point:
                change=min_change
                if pt[0]-poly_left_width>=self.border_left and pt[0]+poly_right_width=<self.border_right:
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
            reference_point=self.polys[GeoFunc.checkTop(self.polys[i])]
            self.cur_polys.append(GeoFunc.getSlide(self.polys[i],target_position[0]-reference_point[0],target_position[1]-reference_point[1]))

        self.moveToBottomLeft()
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
    
    '''该部分还需要完成！！！！！'''
    def chooseFeasiblePoint(self,border):
        '''
        1. 将Polygon对象转化为点
        2. 超出Width范围的点排除
        3. 直线与边界的交点选入
        '''
        res=mapping(border)
        _arr=[]
        if res["type"]=="MultiPolygon":
            for poly in res["coordinates"]:
                for point in poly[0]:
                    _arr.append([point[0],point[1]])
        else:
            for point in res["coordinates"][0]:
                _arr.append([point[0],point[1]])
        
        # (1) 超出了上侧&总宽度没有超过
        # feasible1=pt[1]-border_top[1]>0 and pt[1]-border_top[1]+border_height<=self.width
        # (2) 超过了下侧&总宽度没有超过
        # feasible2=border_bottom[1]-pt[1]>0 and border_bottom[1]-pt[1]+height<=self.width
        # (3) Top和bottom的内部
        # feasible3=pt[1]<=border_bottom[1] and pt[1]>=bottom

        return _arr

    def slideToBottomLeft(self):
        '''移到最左下角位置'''
        for poly in self.cur_polys:
            GeoFunc.slidePoly(poly,-self.border_left,-self.border_bottom)

    def showResult(self):
        '''显示排样结果'''
        for poly in self.cur_polys:
            PltFunc.addPolygon(poly)
        PltFunc.showPolys()

    
if __name__=='__main__':
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