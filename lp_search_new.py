"""
Hybrid Algorithm：Guided Search with Modified Penetration Depth
-----------------------------------
Created on Wed June 10, 2020
@author: seanys,prinway
-----------------------------------
"""
from tools.polygon import PltFunc,getData
from tools.geo_assistant import GeometryAssistant
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

class GSMPD(object):
    """
    Guided Search with Modified Penetration Depth:
    We revised penetration depth to convert searching over layout into a 
    linear programming problem. Based on the guided search methodology, the 
    layout of a nesting problem can be optimzated step by step. Our approach 
    make directly finding the best position for a polygon over layout, which 
    can save the searching time and get better result.
    
    Attention: This file has only adapted to convex polys
    """
    def __init__(self, width, polys):
        self.width = width # 容器的宽度
        self.initialProblem(1) # 获得全部
        self.ration_dec, self.ration_inc = 0.04, 0.01
        self.main()

    def main(self):
        '''核心算法部分'''
        self.cur_length = self.best_length*(1 - self.ration_dec) # 当前的宽度
        self.slideToContainer() # 把突出去的移进来

        self.minimizeOverlap() 

        pass

    def minimizeOverlap(self):
        '''最小化某个重叠情况'''
        self.miu = [[1]*len(self.polys) for _ in range(len(self.polys))] # 计算重叠权重调整（每次都会更新）
        N,it = 50,0 # 记录计算次数
        Fitness = 9999999999999 # 记录Fitness即全部的PD
        while it < N:
            permutation = np.arange(len(self.polys))
            np.random.shuffle(permutation)
            # permutation = [4,7,2,8,3,1,10,0,6,5,9,11]
            for i in range(len(self.polys)):
                choose_index = permutation[i] # 选择形状位置
                top_pt = GeometryAssistant.getTopPoint(self.polys[choose_index]) # 获得最高位置，用于计算PD
                cur_pd = self.getIndexPD(choose_index,top_pt,self.orientation[choose_index]) # 计算某个形状全部pd
                if cur_pd < bias: # 如果没有重叠就直接退出
                    continue
                final_pt, final_pd = copy.deepcopy(top_pt), cur_pd # 记录最佳情况
                for ori in [0]: # 测试全部的方向
                    min_pd,best_pt = self.lpSearch(choose_index,ori) # 为某个形状寻找更优的位置
                    if min_pd < final_pd:
                        final_pd,final_pt = min_pd,copy.deepcopy(best_pt) # 更新高度和计算位置
                if final_pd < cur_pd: # 更新最佳情况
                    GeometryAssistant.slideToPoint(self.polys[choose_index],final_pt) # 平移到目标区域
            total_pd,pd_pair,max_pair_pd = self.getAllPD() # 更新整个计算结果
            if total_pd < bias:
                return
            elif total_pd < Fitness:
                Fitness = total_pd
                it = 0
            self.updateMiu(max_pair_pd,pd_pair)
            it = it + 1

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
            cutted_res = cur_NFP.intersection(IFR)
            cutted_NFPs.append(cutted_res) # 添加切除后的NFP，且不考虑面积过小的
        
        '''如果剩余的面积大于Bias则选择一个点，该阶段不需要考虑在边界的情况'''
        if feasible_IFR.area > bias:
            potential_points = GeometryAssistant.kwtGroupToArray(feasible_IFR,0)
            random_index = random.randint(0, len(potential_points) - 1)
            return 0,potential_points[random_index]
        
        '''计算各个阶段的NFP情况'''
        NFP_stages.append(copy.deepcopy(cutted_NFPs)) # 计算第一次Cut掉的结果
        nfp_neighbor = self.getNFPNeighbor(cutted_NFPs,i) # 获得邻接NFP
        index_stages.append([[i] for i in range(len(cutted_NFPs))]) # 每个阶段对应的重叠区域
        stage_index = 1 # 阶段情况
        while True:
            NFP_stages.append([]) # 增加新的阶段
            index_stages.append([])
            new_last_stage = copy.deepcopy(NFP_stages[stage_index-1]) # 记录上一阶段的结果
            '''枚举上一阶段NFP交集'''
            for k,inter_nfp in enumerate(NFP_stages[stage_index-1]):
                '''获得重叠目标区域，分三种情况'''
                if stage_index > 2:
                    target_indexs = nfp_neighbor[index_stages[stage_index-1][k][-1]] # 仅考虑最新加入的
                elif stage_index == 2:
                    target_indexs = list(set(nfp_neighbor[index_stages[stage_index-1][k][0]] + nfp_neighbor[index_stages[stage_index-1][k][1]]))
                else:
                    target_indexs = nfp_neighbor[index_stages[stage_index-1][k][0]] # 考虑首个形状的
                target_indexs = self.removeItmes(target_indexs,index_stages[stage_index-1][k])
                '''1. 求切除后的差集，可能为Polygon、MultiPolygon等
                   2. 求解和新的形状的交集，并记录在NFP_stages'''
                for poly_index in target_indexs:
                    new_last_stage[k] = new_last_stage[k].difference(cutted_NFPs[poly_index]) # 计算上一阶段的差集
                    inter_region = NFP_stages[stage_index-1][k].intersection(cutted_NFPs[poly_index]) # 计算当前的交集
                    if inter_region.area > 0 and index_stages[stage_index-1][k][-1] < poly_index: # 如果不是空集
                        NFP_stages[stage_index].append(inter_region) # 记录到各阶段的NFP
                        index_stages[stage_index].append(index_stages[stage_index-1][k] + [poly_index]) # 记录到各阶段的NFP
                NFP_stages[stage_index-1] = copy.deepcopy(new_last_stage) # 更新上一阶段结果
            '''结束条件'''
            if len(NFP_stages[stage_index]) == 0:
                break
            stage_index = stage_index + 1
        
        """遍历全部的点计算最佳情况"""
        min_pd,total_num_pt,best_pt = 99999999999,0,[]
        for k,stage in enumerate(NFP_stages):
            for w,item in enumerate(stage):
                if item.is_empty == True: # 判断空集
                    continue
                '''获得全部的参考点，并寻找最佳位置'''
                all_pts = self.getAllPoint(item)
                '''计算当前的pd'''
                for pt in all_pts:
                    total_pd = 0 # 初始的值
                    for poly_index in index_stages[k][w]:
                        pd = self.getPtNFPPD(pt,basic_nfps[poly_index])
                        total_pd = total_pd + pd*self.miu[i][poly_index] # 计算全部的pd
                    if total_pd < min_pd:
                        min_pd = total_pd
                        best_pt = [pt[0],pt[1]] # 更新最佳位置
                    total_num_pt = total_num_pt + 1 # 增加检索位置
        # GeometryAssistant.slideToPoint(self.polys[i],best_pt) # 平移到目标区域
        # self.showPolys()
        return min_pd,best_pt

    def getAllPD(self):
        '''获得当前全部形状间的PD'''
        total_pd,pd_pair,max_pair_pd = 0,[[0]*len(self.polys) for _ in range(len(self.polys))],0 # 两两之间的pd和总的pd
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                pd = self.getPolysPD(i,j)
                pd_pair[i][j],pd_pair[j][i] = pd,pd # 更新对应的pd
                total_pd = total_pd + pd # 更新总的pd
                if pd > max_pair_pd: # 更新最大的值
                    max_pair_pd = pd
        return total_pd,pd_pair,max_pair_pd

    def getPolysPD(self, i, j):
        '''获得两个形状间的PD，根据点计算'''
        top_pt = GeometryAssistant.getTopPoint(self.polys[i])
        nfp = self.getNFP(i, j, self.orientation[i], self.orientation[j])
        if Polygon(nfp).contains(Point(top_pt)) == True:
            return self.getPtNFPPD(top_pt,nfp)
        else:
            return 0

    def getIndexPD(self,i,top_pt,oi):
        '''获得某个形状的PD'''
        total_pd,target = 0, [j for j in range(len(self.polys)) if j != i] # 获得全部的NFP基础
        for j in target:
            nfp = self.getNFP(i, j, oi, self.orientation[j]) # 获得NFP结果
            if Polygon(nfp).contains(Point(top_pt)) == True: # 包含的情况下才计算
                total_pd = total_pd + self.getPtNFPPD(top_pt,nfp) # 计算PD，比较简单
        return total_pd

    def getPtNFPPD(self, pt, nfp):
        '''根据顶点和nfp计算PD'''
        min_pd,min_edge = 999999999999,[]
        edges = GeometryAssistant.getPolyEdges(nfp)
        for edge in edges:
            A = edge[0][1] - edge[1][1]
            B = edge[1][0] - edge[0][0]
            C = edge[0][0]*edge[1][1] - edge[1][0]*edge[0][1]
            D = math.pow(A*A + B*B,0.5)
            a,b,c = A/D,B/D,C/D
            if abs(a*pt[0] + b*pt[1] + c) < min_pd:
                min_pd,min_edge = abs(a*pt[0] + b*pt[1] + c),copy.deepcopy(edge)
        return min_pd

    def updateMiu(self,max_pair_pd,pd_pair):
        """寻找到更优位置之后更新"""
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                self.miu[i][j] = self.miu[i][j] + pd_pair[i][j]/max_pair_pd
                self.miu[j][i] = self.miu[j][i] + pd_pair[j][i]/max_pair_pd
    
    def removeItmes(self,input_list,del_target):
        '''删除中间的元素，不考虑计算速度'''
        new_input_list = copy.deepcopy(input_list)
        for item in del_target:
            new_input_list = [i for i in new_input_list if i != item]
        return new_input_list
    
    def addTuplePoly(self,_arr,_tuple):
        """增加tuple格式的多边形，不添加最后一个"""
        for i,pt in enumerate(_tuple):
            if i == len(_tuple) - 1 :
                break
            _arr.append([pt[0],pt[1]])
    
    def getAllPoint(self,item):
        """获得某个形状对应的全部点"""
        all_pts = []
        if item.geom_type == "Polygon":
            self.addTuplePoly(all_pts,mapping(item)["coordinates"][0])
        elif item.geom_type == "MultiPolygon":
            for w,sub_item in enumerate(mapping(item)["coordinates"]):
                self.addTuplePoly(all_pts,sub_item[0])                    
        else:
            self.outputWarning("出现未知几何类型")
        return all_pts

    def slideToContainer(self):
        '''将超出的多边形左移'''
        for index,poly in enumerate(self.polys):
            right_pt = GeometryAssistant.getRightPoint(poly)
            if right_pt[0] > self.cur_length:
                delta_x = self.cur_length-right_pt[0]
                GeometryAssistant.slidePoly(poly,delta_x,0)
    
    def getNFPNeighbor(self, NFPs, index):
        '''获得NFP之间的重叠列表，只存比自己序列更大的'''
        nfp_neighbor = [[] for _ in range(len(NFPs))]
        for i in range(len(NFPs)-1):
            for j in range(i+1, len(NFPs)):
                if i == index or j == index:
                    continue
                if NFPs[i].intersects(NFPs[j]) == True:
                    nfp_neighbor[i].append(j)     
                    nfp_neighbor[j].append(i)     
        return nfp_neighbor

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

    def outputWarning(self,_str):
        print("\033[0;31m%s\033[0m" % _str)

    def outputAttention(self,_str):
        print("\033[0;32m%s\033[0m" % _str)

if __name__=='__main__':
    polys=getData()
    GSMPD(760,polys)
