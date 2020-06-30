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
import operator
import multiprocessing

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
        self.initialProblem(12) # 获得全部
        self.ration_dec, self.ration_inc = 0.04, 0.01
        self.TEST_MODEL = False
        # self.showPolys()
        self.main()

    def main(self):
        '''核心算法部分'''
        print("初始利用率为：",433200/(self.cur_length*self.width))
        # self.showPolys()
        # return 
        self.shrinkBorder() # 平移边界并更新宽度
        # self.extendBorder()

        max_time = 200000
        if self.TEST_MODEL == True:
            max_time = 3

        start_time = time.time()
        while time.time() - start_time < max_time:
            self.intialPairPD() # 初始化当前两两间的重叠
            feasible = self.minimizeOverlap() # 开始最小化重叠
            if feasible == True:
                print("当前利用率为：",433200/(self.cur_length*self.width))
                self.best_orientation = copy.deepcopy(self.orientation) # 更新方向
                self.best_polys = copy.deepcopy(self.polys) # 更新形状
                self.best_length = self.cur_length # 更新最佳高度
                with open("/Users/sean/Documents/Projects/Packing-Algorithm/record/lp_result.csv","a+") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[time.asctime( time.localtime(time.time()) ),feasible,self.best_length,433200/(self.best_length*self.width),self.orientation,self.polys]])
                # self.showPolys()
                self.shrinkBorder() # 收缩边界并平移形状到内部来
            else:
                self.outputWarning("结果不可行，重新进行检索")
                with open("/Users/sean/Documents/Projects/Packing-Algorithm/record/lp_result.csv","a+") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[time.asctime( time.localtime(time.time()) ),feasible,self.best_length,433200/(self.best_length*self.width),self.orientation,self.polys]])        
                self.extendBorder() # 扩大边界并进行下一次检索

    def minimizeOverlap(self):
        '''最小化某个重叠情况'''
        self.miu = [[1]*len(self.polys) for _ in range(len(self.polys))] # 计算重叠权重调整（每次都会更新）
        N,it = 50,0 # 记录计算次数
        Fitness = 9999999999999 # 记录Fitness即全部的PD
        if self.TEST_MODEL == True: # 测试模式
            N = 1
        self.changed_status = True # True表示修改过，False表示没有修改过
        while it < N:
            # changed = False # 该参数表示是否修改过
            print("第",it,"轮")
            permutation = np.arange(len(self.polys))
            np.random.shuffle(permutation)
            # print(permutation)
            # permutation = [7,4,2,6,10,9,0,1,3,5,11,8]
            for i in range(len(self.polys)):
                choose_index = permutation[i] # 选择形状位置
                top_pt = GeometryAssistant.getTopPoint(self.polys[choose_index]) # 获得最高位置，用于计算PD
                cur_pd = self.getIndexPD(choose_index,top_pt,self.orientation[choose_index]) # 计算某个形状全部pd
                if cur_pd < bias: # 如果没有重叠就直接退出
                    continue
                final_pt, final_pd, final_ori = copy.deepcopy(top_pt), cur_pd, self.orientation[choose_index] # 记录最佳情况
                
                # tasks = [[[choose_index,ori]] for ori in [0,1,2,3]]
                # pool = multiprocessing.Pool(4)
                # results = pool.starmap(self.lpSearch,tasks)
                # for [min_pd,best_pt,ori] in results:
                #     if min_pd < final_pd:
                #         final_pd,final_pt,final_ori = min_pd,copy.deepcopy(best_pt),ori # 更新高度，位置和方向
                for ori in [0,1,2,3]: # 测试全部的方向
                    min_pd,best_pt = self.lpSearch(choose_index,ori) # 为某个形状寻找更优的位置
                    if min_pd < final_pd:
                        final_pd,final_pt,final_ori = min_pd,copy.deepcopy(best_pt),ori # 更新高度，位置和方向
                if final_pd < cur_pd: # 更新最佳情况
                    print(choose_index,"寻找到更优位置:",cur_pd,"->",final_pd)
                    self.polys[choose_index] = copy.deepcopy(self.all_polygons[choose_index][final_ori]) # 形状对象
                    GeometryAssistant.slideToPoint(self.polys[choose_index],final_pt) # 平移到目标区域
                    self.orientation[choose_index] = final_ori # 更新方向
                    # print(self.pair_pd_record)
                    self.updatePD(choose_index) # 更新对应元素的PD，线性时间复杂度
                    # print(self.pair_pd_record)
                    # self.changed_status,changed = True,True
                else:
                    print(choose_index,"未寻找到更优位置")
            total_pd,max_pair_pd = self.getPDStatus() # 获得当前的PD情况
            if total_pd < bias:
                self.outputWarning("结果可行")                
                return True
            elif total_pd < Fitness:
                Fitness = total_pd
                it = 0
                _str = "更新最少重叠:" + str(total_pd)
                self.outputAttention(_str)

            # self.changed_status = changed # 记录该轮是否修改了
            self.updateMiu(max_pair_pd) # 更新参数
            it = it + 1 # 更新计数次数

            _str = "当前全部重叠:" + str(total_pd)
            self.outputInfo(_str) # 输出当前重叠情况
        return False
    
    def lpSearchRecord(self, i, oi):
        '''基于历史情况进行更新'''
        print("快捷检索:",i,oi)
        pt_to_polys = self.pt_to_polys_record[i][oi] # 该形状所有的点对应的形状编号
        min_pd,total_num_pt,best_pt = 99999999999,0,[]
        for target in pt_to_polys:
            total_pd = 0
            for target_index in target[2]:
                pd = self.pd_pt_poly_record[i][oi][target_index] # 与多边形的情况
                total_pd = total_pd + pd * self.miu[i][poly_index] # 计算目标形状与其他形状调整后的全部的pd
            if total_pd < min_pd:
                min_pd = total_pd
                best_pt = [target[0],target[1]] # 更新最佳位置
        return min_pd,best_pt

    # def lpSearch(self, target_status):
    def lpSearch(self, i, oi):
        '''
        为某个形状的某个角度寻找最佳位置：输入形状和角度，根据miu调整后的情况，
        找到综合的穿透深度最低的位置。
        过程：首先计算IFR和全部的NFP (1) 计算NFP切除IFR和IFR切除全部NFP后
        的结果，如果IFR切除后仍然有空余，就随机选择一个点返回 (2)如果IFR切除
        后没有可行点，则计算NFP切除的各个阶段的点，同时通过NFP的Point和Edge
        计算最佳情况，寻找最佳位置
        '''
        # [i, oi] = target_status
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
            # return [0,potential_points[random_index],oi]
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
                    try:
                        new_last_stage[k] = new_last_stage[k].difference(cutted_NFPs[poly_index]) # 计算上一阶段的差集
                        inter_region = NFP_stages[stage_index-1][k].intersection(cutted_NFPs[poly_index]) # 计算当前的交集
                    except BaseException:
                        inter_region = Polygon([])
                        self.outputWarning("出现几何计算错误")
                    if inter_region.area > 0 and index_stages[stage_index-1][k][-1] < poly_index: # 如果不是空集
                        NFP_stages[stage_index].append(inter_region) # 记录到各阶段的NFP
                        index_stages[stage_index].append(index_stages[stage_index-1][k] + [poly_index]) # 记录到各阶段的NFP
                try:
                    NFP_stages[stage_index-1] = copy.deepcopy(new_last_stage) # 更新上一阶段结果
                except BaseException:
                    self.outputWarning("出现复制错误")
            '''结束条件'''
            if len(NFP_stages[stage_index]) == 0:
                break
            stage_index = stage_index + 1
        
        '''遍历和记录全部的点及其对应区域'''
        all_search_targets = []
        for k,stage in enumerate(NFP_stages):
            for w,item in enumerate(stage):
                if item.is_empty == True: # 判断空集
                    continue
                new_pts = self.getAllPoint(item) # 获得全部的点
                for pt in new_pts: # 增加记录全部情况
                    all_search_targets.append([pt[0],pt[1],index_stages[k][w]])
        all_search_targets = sorted(all_search_targets, key=operator.itemgetter(0, 1)) # 按照升序进行排列
        
        '''求解出全部的点的最佳情况'''
        min_pd,total_num_pt,best_pt = 99999999999,0,[]
        for k,target in enumerate(all_search_targets):
            if target[0] == all_search_targets[k-1][0] and target[1] == all_search_targets[k-1][1]:
                continue
            total_pd = 0 # 初始的值
            for poly_index in target[2]:
                pd = self.getPtNFPPD([target[0],target[1]],basic_nfps[poly_index])
                total_pd = total_pd + pd * self.miu[i][poly_index] # 计算目标形状与其他形状调整后的全部的pd
            if total_pd < min_pd:
                min_pd = total_pd
                best_pt = [target[0],target[1]] # 更新最佳位置
            total_num_pt = total_num_pt + 1
        
        return min_pd,best_pt
        # return [min_pd,best_pt,oi]

    def updatePD(self,choose_index):
        '''平移某个元素后更新对应的PD'''
        for i in range(len(self.polys)):
            if i == choose_index:
                continue
            pd = self.getPolysPD(choose_index,i)
            self.pair_pd_record[i][choose_index],self.pair_pd_record[choose_index][i] = pd,pd # 更新对应的pd

    def getPDStatus(self):
        '''获得当前的最佳情况'''
        total_pd,max_pair_pd = 0,0
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                total_pd = total_pd + self.pair_pd_record[i][j]
                if self.pair_pd_record[i][j] > max_pair_pd:
                    max_pair_pd = self.pair_pd_record[i][j]
        return total_pd,max_pair_pd

    def intialPairPD(self):
        '''获得当前全部形状间的PD，无需调整'''
        self.pair_pd_record = [[0]*len(self.polys) for _ in range(len(self.polys))] # 两两之间的pd和总的pd
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                pd = self.getPolysPD(i,j)
                self.pair_pd_record[i][j],self.pair_pd_record[j][i] = pd,pd # 更新对应的pd

    def getPolysPD(self, i, j):
        '''获得当前两个形状间的PD，根据点计算，无需调整'''
        top_pt = GeometryAssistant.getTopPoint(self.polys[i])
        nfp = self.getNFP(i, j, self.orientation[i], self.orientation[j])
        if Polygon(nfp).contains(Point(top_pt)) == True:
            return self.getPtNFPPD(top_pt,nfp)
        else:
            return 0

    def getIndexPD(self,i,top_pt,oi):
        '''获得某个形状的全部PD，是调整后的结果'''
        total_pd = 0 # 获得全部的NFP基础
        for j in range(len(self.polys)):
            total_pd = total_pd + self.pair_pd_record[i][j]*self.miu[i][j] # 计算PD，比较简单
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

    def updateMiu(self,max_pair_pd):
        """寻找到更优位置之后更新"""
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                self.miu[i][j] = self.miu[i][j] + self.pair_pd_record[i][j]/max_pair_pd
                self.miu[j][i] = self.miu[j][i] + self.pair_pd_record[j][i]/max_pair_pd
        # print("miu:",self.miu)
    
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
                try: # 可能会报错
                    self.addTuplePoly(all_pts,sub_item[0])
                except BaseException:
                    self.outputWarning(sub_item)
        elif item.geom_type == "GeometryCollection":
            try: # 可能会报错
                for w,sub_item in enumerate(mapping(item)["geometries"]):
                    if sub_item["type"] == "Polygon":
                        self.addTuplePoly(all_pts,sub_item[0])
            except BaseException:
                self.outputWarning(item)
        else:
            pass # 只有直线一种情况，不影响计算结果
        return all_pts

    def shrinkBorder(self):
        '''收缩边界，将超出的多边形左移'''
        self.cur_length = self.best_length*(1 - self.ration_dec)
        for index,poly in enumerate(self.polys):
            right_pt = GeometryAssistant.getRightPoint(poly)
            if right_pt[0] > self.cur_length:
                delta_x = self.cur_length-right_pt[0]
                GeometryAssistant.slidePoly(poly,delta_x,0)
    
    def extendBorder(self):
        '''扩大边界'''
        self.cur_length = self.best_length*(1 + self.ration_inc)

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
        row = j*192 + i*16 + oj*4 + oi # i为移动形状，j为固定位置
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
        '''输出红色字体'''
        print("\033[0;31m",_str,"\033[0m")

    def outputAttention(self,_str):
        '''输出绿色字体'''
        print("\033[0;32m",_str,"\033[0m")

    def outputInfo(self,_str):
        '''输出浅黄色字体'''
        print("\033[0;33m",_str,"\033[0m")

if __name__=='__main__':
    polys=getData()
    GSMPD(760,polys)
