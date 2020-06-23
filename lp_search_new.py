"""
Hybrid Algorithm：Guided Search with Modified Penetration Depth
-----------------------------------
Created on Wed June 10, 2020
@author: seanys,prinway
-----------------------------------
"""
from tools.polygon import PltFunc,getData
from shapely.geometry import Polygon,Point,mapping,LineString
import pandas as pd
import json
import copy
import random
import math
import datetime
import time
import csv # 写csv
import numpy as np

bias=0.0000001

class GeometryAssistant(object):
    '''
    几何相关的算法重新统一
    '''
    @staticmethod
    def getPolysRight(polys):
        _max=0
        for i in range(0,len(polys)):
            [x,y] = GeometryAssistant.getRightPoint(polys[i])
            if x>_max:
                _max=x
        return _max
    
    @staticmethod
    def kwtGroupToArray(kwt_group, judge_area):
        '''将几何对象转化为数组，以及是否判断面积大小'''
        array = []
        if kwt_group.geom_type == "Polygon":
            array = GeometryAssistant.kwtItemToArray(region, judge_area)  # 最终结果只和顶点相关
        else:
            for shapely_item in list(kwt_group):
                array = array + GeometryAssistant.kwtItemToArray(shapely_item)
        return area   

    @staticmethod
    def kwtItemToArray(kwt_item, judge_area):
        '''将一个kwt对象转化为数组（比如Polygon）'''
        if judge_area == True and kwt_item.area < bias:
            return []
        res = mapping(kwt_item)
        _arr = []
        # 去除重叠点的情况
        if res["coordinates"][0][0] == res["coordinates"][0][-1]:
            for point in res["coordinates"][0][0:-1]:
                _arr.append([point[0],point[1]])
        else:
            for point in res["coordinates"][0]:
                _arr.append([point[0],point[1]])
        return _arr

    @staticmethod
    def getPolyEdges(poly):
        edges = []
        for index,point in enumerate(poly):
            if index < len(poly)-1:
                edges.append([poly[index],poly[index+1]])
            else:
                edges.append([poly[index],poly[0]])
        return edges

    @staticmethod
    def getInnerFitRectangle(poly,x,y):
        left_pt, bottom_pt, right_pt, top_pt = GeometryAssistant.getBoundPoint(poly) # 获得全部边界点
        intial_pt = [top_pt[0] - left_pt[0], top_pt[1] - bottom_pt[1]] # 计算IFR初始的位置
        ifr_width = x - right_pt[0] + left_pt[0]  # 获得IFR的宽度
        ifr = [[intial_pt[0], intial_pt[1]], [intial_pt[0] + ifr_width, intial_pt[1]], [intial_pt[0] + ifr_width, y], [intial_pt[0], y]]
        return ifr
    
    @staticmethod
    def getSlide(poly,x,y):
        '''获得平移后的情况'''
        new_vertex=[]
        for point in poly:
            new_point = [point[0]+x,point[1]+y]
            new_vertex.append(new_point)
        return new_vertex

    @staticmethod
    def slidePoly(poly,x,y):
        '''将对象平移'''
        for point in poly:
            point[0] = point[0] + x
            point[1] = point[1] + y

    @staticmethod
    def getDirectionalVector(vec):
        _len=math.sqrt(vec[0]*vec[0]+vec[1]*vec[1])
        return [vec[0]/_len,vec[1]/_len]

    @staticmethod
    def deleteOnline(poly):
        '''删除两条直线在一个延长线情况'''
        new_poly=[]
        for i in range(-2,len(poly)-2):
            vec1 = GeometryAssistant.getDirectionalVector([poly[i+1][0]-poly[i][0],poly[i+1][1]-poly[i][1]])
            vec2 = GeometryAssistant.getDirectionalVector([poly[i+2][0]-poly[i+1][0],poly[i+2][1]-poly[i+1][1]])
            if abs(vec1[0]-vec2[0])>bias or abs(vec1[1]-vec2[1])>bias:
                new_poly.append(poly[i+1])
        return new_poly

    @staticmethod
    def getTopPoint(poly):
        top_pt,max_y=[],-999999999
        for pt in poly:
            if pt[1]>max_y:
                max_y=pt[1]
                top_pt=[pt[0],pt[1]]
        return top_pt

    @staticmethod
    def getBottomPoint(poly):
        bottom_pt,min_y=[],999999999
        for pt in poly:
            if pt[1]<min_y:
                min_y=pt[1]
                bottom_pt=[pt[0],pt[1]]
        return bottom_pt

    @staticmethod
    def getRightPoint(poly):
        right_pt,max_x=[],-999999999
        for pt in poly:
            if pt[0]>max_x:
                max_x=pt[0]
                right_pt=[pt[0],pt[1]]
        return right_pt

    @staticmethod
    def getLeftPoint(poly):
        left_pt,min_x=[],999999999
        for pt in poly:
            if pt[0]<min_x:
                min_x=pt[0]
                left_pt=[pt[0],pt[1]]
        return left_pt

    @staticmethod
    def getBottomLeftPoint(poly):
        bottom_left_pt,min_x,min_y=[],999999999,999999999
        for pt in poly:
            if pt[0]<=min_x and pt[1]<=min_y:
                min_x,min_y=pt[0],pt[1]
                bottom_left_pt=[pt[0],pt[1]]
        return bottom_left_pt

    @staticmethod
    def getBoundPoint(poly):
        left_pt,bottom_pt,right_pt,top_pt=[],[],[],[]
        min_x,min_y,max_x,max_y=999999999,999999999,-999999999,-999999999
        for pt in poly:
            if pt[0]<min_x:
                min_x=pt[0]
                left_pt=[pt[0],pt[1]]
            if pt[0]>max_x:
                max_x=pt[0]
                right_pt=[pt[0],pt[1]]
            if pt[1]>max_y:
                max_y=pt[1]
                top_pt=[pt[0],pt[1]]
            if pt[1]<min_y:
                min_y=pt[1]
                bottom_pt=[pt[0],pt[1]]
        return left_pt,bottom_pt,right_pt,top_pt

class GSMPD(object):
    """
    Guided Search with Modified Penetration Depth:
    We revised penetration depth to convert searching over layout into a 
    linear programming problem. Based on the guided search methodology, the 
    layout of a nesting problem can be optimzated step by step. Our approach 
    make directly finding the best position for a polygon over layout, which 
    can save the searching time and get better result.
    """
    def __init__(self, width, polys):
        self.width = width # 容器的宽度
        self.initialProblem(1) # 获得全部
        self.ration_dec, self.ration_inc = 0.04, 0.01
        self.main()

    def main(self):
        '''核心算法部分'''
        # self.cur_length = self.best_length*(1 - self.ration_dec) # 当前的宽度
        self.cur_length = self.best_length
        self.slideToContainer() # 把突出去的移进来

        self.minimizeOverlap()

        pass

    def minimizeOverlap(self):
        '''
        最小化某个重叠情况
        '''
        self.miu = [[[1]*12]*12] # 计算重叠权重调整
        pt = self.lpSearch(11,0) # 为某个形状寻找更优的位置
        print("检索到可行位置", pt)
        # self.showPolys()

    def lpSearch(self, i, oi):
        '''
        为某个形状的某个角度寻找最佳位置：输入形状和角度，根据miu调整后的情况，
        找到综合的穿透深度最低的位置。
        过程：首先计算IFR和全部的NFP (1) 计算NFP切除IFR和IFR切除全部NFP后
        的结果，如果IFR切除后仍然有空余，就随机选择一个点返回 (2)如果IFR切除
        后没有可行点，则计算NFP切除的各个阶段的点，同时通过NFP的Point和Edge
        计算最佳情况，寻找最佳位置
        '''
        polygon = self.getPolygon(i, oi) # 获得对应的形状
        ifr = GeometryAssistant.getInnerFitRectangle(polygon, self.cur_length, self.width)
        IFR, feasible_IFR = Polygon(ifr), Polygon(ifr) # 全部的IFR和可行的IFR
        cutted_NFPs ,NFP_stages, index_stages = [], [], [] # 获得切除后的NFP、各个阶段的NFP（全部是几何对象）、各个阶段对应的NFP情况
        basic_nfps = [] # 获得全部的NFP基础
    
        '''获得全部的NFP以及切除后的NFP情况'''
        for j in range(len(self.polys)):
            if j == i:
                basic_nfps.append([])
                cutted_NFPs.append(Polygon([]))
                continue
            basic_nfps.append(self.getNFP(i, j, oi, self.orientation[j])) # 添加NFP
            cur_NFP = Polygon(basic_nfps[j]) # 获得NFP的Polygon格式，进行切除和相交计算
            feasible_IFR = feasible_IFR.difference(cur_NFP) # 求解可行区域
            cutted_res = cur_NFP.difference(IFR)
            if cutted_res.area > bias:
                cutted_NFPs.append(cutted_res) # 添加切除后的NFP，且不考虑面积过小的
            else:
                cutted_NFPs.append(Polygon([])) # 否则添加空的
        
        if feasible_IFR.area > bias:
            potential_points = GeometryAssistant.kwtGroupToArray(feasible_IFR)
            random_index = random.randint(0, len(potential_points) - 1)
            return potential_points[random_index]
        
        '''计算各个阶段的NFP情况'''
        NFP_stages.append(copy.deepcopy(cutted_NFPs)) # 计算第一次Cut掉的结果
        nfp_neighbor = self.getNFPNeighbor(cutted_NFPs,i) # 获得邻接NFP
        index_stages.append([[i] for i in range(len(NFPs))]) # 第一阶段各个对应的区域
        stage_index = 1 # 阶段情况
        while True:
            NFP_stages.append([]) # 增加新的阶段
            indep_stages = [] # 存储上一个阶段的切除后的情况
            for n in range(len(NFP_stages[stage_index-1])): # 枚举上一阶段所有内容，将其与对应的NFP计算交集
                for m in index_stages[stage_index-1][n][-1]: # 枚举上一阶段最后的
                    pass
        
        '''计算每个NFP用于计算深度的参数'''
        pd_coef = []
        for j in range(len(self.polys)):
            if j == i:
                pd_coef.append([])
                continue
            edges = GeometryAssistant.getPolyEdges(basic_nfps[j])
            # 首先计算所有的边
            for k,edge in enumerate(edges):
                pass
            # 然后计算所有的顶点
            for k,edge in enumerate(edges):
                pass

    def getAllPD(self):
        '''获得当前全部形状间的PD'''
        pass

    def getPolysPD(self, i, j):
        '''获得两个形状间的PD，根据点计算'''
        pass

    def getPtNFPPD(self, pt, nfp):
        '''根据顶点和nfp计算PD'''
        pass

    def getOverlapStatus(self):
        self.overlap_status = []
        for i in range(len(self.polys)):
            for j in range(i + 1, len(self.polys)):
                

    def slideToContainer(self):
        '''将超出的多边形左移'''
        for index,poly in enumerate(self.polys):
            right_pt = GeometryAssistant.getRightPoint(poly)
            if right_pt[0] > self.cur_length:
                delta_x = self.cur_length-right_pt[0]
                GeometryAssistant.slidePoly(poly,delta_x,0)
    
    def getNFPNeighbor(NFPs, index):
        '''获得NFP之间的重叠列表，只存比自己序列更大的'''
        nfp_neighbor = [[] for _ in range(len(NFPs))]
        for i in range(len(NFPs)):
            for j in range(i, len(NFPs)):
                if i == index or j == index:
                    continue
                if NFPs[i].intersection(NFPs[j]).area > bias:
                    nfp_neighbor[i].append(j)                

    def getNFP(self, i, j, oi, oj):
        '''根据形状和角度获得NFP的情况'''
        row = j*192 + i*16 + oi*4 + oj # i为移动形状，j为固定位置
        bottom_pt = GeometryAssistant.getBottomPoint(self.polys[j])
        nfp = GeometryAssistant.getSlide(json.loads(self.all_nfps["nfp"][row]), bottom_pt[0], bottom_pt[1])
        return GeometryAssistant.deleteOnline(nfp) # 需要删除同直线的情况

    def getPolygon(self, index, orientation):
        '''获得某个形状'''
        return self.all_polygons[index][orientation]

    def initialProblem(self, index):
        '''获得某个解，基于该解进行优化'''
        _input = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/lp.csv")
        self.polys, self.best_polys = json.loads(_input["polys"][index]), json.loads(_input["polys"][index]) # 获得形状
        self.orientation, self.best_orientation = json.loads(_input["orientation"][index]),json.loads(_input["orientation"][index]) # 当前的形状状态（主要是角度）
        self.total_area = _input["total_area"][index] # 用来计算利用率
        self.use_ratio = [] # 记录利用率
        self.getPreData() # 获得NFP和全部形状
        self.cur_length = GeometryAssistant.getPolysRight(self.polys) # 获得当前高度
        self.best_length = self.cur_length # 最佳高度
        print("一共",len(self.polys),"个形状")

    def getPreData(self):
        '''获得全部的NFP和各个方向的形状'''
        self.all_polygons = [] # 存储所有的形状及其方向
        fu = pd.read_csv("/Users/sean/Documents/Projects/Data/fu_orientation.csv") 
        for i in range(fu.shape[0]):
            polygons=[]
            for j in ["o_0","o_1","o_2","o_3"]:
                polygons.append(json.loads(fu[j][i]))
            self.all_polygons.append(polygons)
        self.all_nfps = pd.read_csv("/Users/sean/Documents/Projects/Data/fu.csv") # 获得NFP

    def showPolys(self):
        '''展示全部形状以及边框'''
        for poly in self.polys:
            PltFunc.addPolygon(poly)
        PltFunc.addPolygonColor([[0,0], [self.cur_length,0], [self.cur_length,self.width], [0,self.width]])
        PltFunc.showPlt(width=1000, height=1000)

    def showPolys(self):
        for poly in self.polys:
            PltFunc.addPolygon(poly)
        PltFunc.addPolygonColor([[0,0],[self.cur_length,0],[self.cur_length,self.width],[0,self.width]])
        PltFunc.showPlt(width=1000,height=1000)

if __name__=='__main__':
    polys=getData()
    GSMPD(760,polys)
    # print()