"""
该版本开始优化性能，尽量全部采用直接几何计算
-----------------------------------
Created on Wed June 10, 2020
@author: seanys,prinway
-----------------------------------
"""
from tools.polygon import PltFunc,getData
from tools.geo_assistant import GeometryAssistant, OutputFunc
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

compute_bias = 0.00001
bias = 0.5
max_overlap = 5



class GSMPD(object):
    def __init__(self):
        self.initialProblem(2) # 获得全部 
        self.ration_dec, self.ration_inc = 0.04, 0.01
        self.TEST_MODEL = False
        # self.showPolys()
        self.main()

    def main(self):
        '''核心算法部分'''
        _str = "初始利用率为：" + str(self.total_area/(self.cur_length*self.width))
        OutputFunc.outputAttention(self.set_name,_str)
        self.shrinkBorder() # 平移边界并更新宽度
        max_time = 360000
        if self.TEST_MODEL == True:
            max_time = 1
        self.start_time = time.time()
        search_status = 0
        while time.time() - self.start_time < max_time:
            self.initialPairPD() # 初始化当前两两间的重叠
            feasible = self.minimizeOverlap() # 开始最小化重叠
            # self.showPolys(saving=True)
            if feasible == True:
                search_status = 0
                _str = "当前利用率为：" + str(self.total_area/(self.cur_length*self.width))
                OutputFunc.outputInfo(self.set_name,_str)
                self.best_orientation = copy.deepcopy(self.orientation) # 更新方向
                self.best_polys = copy.deepcopy(self.polys) # 更新形状
                self.best_length = self.cur_length # 更新最佳高度
                with open("record/lp_result/" + self.set_name + "_result_success.csv","a+") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[time.asctime( time.localtime(time.time()) ),feasible,self.best_length,self.total_area/(self.best_length*self.width),self.orientation,self.polys]])
                self.showPolys()
                self.shrinkBorder() # 收缩边界并平移形状到内部来
            else:
                OutputFunc.outputWarning(self.set_name, "结果不可行，重新进行检索")
                with open("record/lp_result/" + self.set_name + "_result_fail.csv","a+") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[time.asctime( time.localtime(time.time()) ),feasible,self.cur_length,self.total_area/(self.cur_length*self.width),self.orientation,self.polys]])        
                if search_status == 1:
                    self.shrinkBorder()
                    search_status = 0
                else:
                    self.extendBorder() # 扩大边界并进行下一次检索
                    search_status = 1    
            if self.total_area/(self.best_length*self.width) > 0.995:
                break

    def minimizeOverlap(self):
        '''最小化某个重叠情况'''
        self.miu = [[1]*len(self.polys) for _ in range(len(self.polys))] # 计算重叠权重调整（每次都会更新）
        N,it = 50,0 # 记录计算次数
        Fitness = 9999999999999 # 记录Fitness即全部的PD
        if self.TEST_MODEL == True: # 测试模式
            N = 1
        unchange_times,last_pd = 0,0
        while it < N:
            print("第",it,"轮")
            permutation = np.arange(len(self.polys))
            np.random.shuffle(permutation)
            # print(permutation)
            # permutation = [j for j in range(len(self.polys))]
            for choose_index in permutation:
                top_pt = GeometryAssistant.getTopPoint(self.polys[choose_index]) # 获得最高位置，用于计算PD
                cur_pd = self.getIndexPD(choose_index,top_pt,self.orientation[choose_index]) # 计算某个形状全部pd
                if cur_pd < self.bias: # 如果没有重叠就直接退出
                    continue
                final_pt, final_ori = copy.deepcopy(top_pt), self.orientation[choose_index] # 记录最佳情况                
                final_pd, final_pd_record = cur_pd, [0 for _ in range(len(self.polys))]
                for ori in self.allowed_rotation: # 测试全部的方向
                    if (ori == 2 and self.vertical_symmetrical[choose_index] == 1) or (ori == 3 and self.horizon_symmetrical[choose_index] == 1):
                        continue
                    min_pd,best_pt,best_pd_record = self.lpSearch(choose_index,ori) # 为某个形状寻找最优的位置
                    if min_pd < final_pd:
                        final_pt,final_ori = copy.deepcopy(best_pt),ori # 更新位置和方向
                        final_pd,final_pd_record = min_pd, copy.deepcopy(best_pd_record) # 记录最终的PD和对应的PD值
                    if min_pd == 0:
                        break
                if final_pd < cur_pd: # 更新最佳情况
                    print(choose_index,"寻找到更优位置",final_pt,":",cur_pd,"->",final_pd)
                    self.polys[choose_index] = self.getPolygon(choose_index,final_ori)
                    GeometryAssistant.slideToPoint(self.polys[choose_index],final_pt) # 平移到目标区域
                    self.orientation[choose_index] = final_ori # 更新方向
                    self.updatePD(choose_index, final_pd_record) # 更新对应元素的PD，线性时间复杂度
                else:
                    print(choose_index,"未寻找到更优位置")
            if self.TEST_MODEL == True: # 测试模式
                return
            # self.showPolys()
            total_pd,max_pair_pd = self.getPDStatus() # 获得当前的PD情况
            if total_pd < self.max_overlap:
                OutputFunc.outputWarning(self.set_name,"结果可行")                
                return True
            elif total_pd < Fitness:
                Fitness = total_pd
                it = 0
                _str = "更新最少重叠:" + str(total_pd)
                OutputFunc.outputAttention(self.set_name, _str)

            self.updateMiu(max_pair_pd) # 更新参数
            it = it + 1 # 更新计数次数

            _str = "当前全部重叠:" + str(total_pd)
            OutputFunc.outputInfo(self.set_name,_str) # 输出当前重叠情况

        return False

    def lpSearch(self, i, oi):
        polygon = self.getPolygon(i, oi) # 获得对应的形状
        ifr, ifr_bounds = GeometryAssistant.getIFRWithBounds(polygon, self.cur_length, self.width)
        ifr_edges, feasible_IFR = GeometryAssistant.getPolyEdges(ifr), Polygon(ifr) # 全部的IFR和可行的IFR
        cur_nfps, cur_nfps_edges, cur_nfps_bounds, cur_convex_status = [], [], [], []

        '''获得全部的NFP及切除后结果'''
        for j in range(len(self.polys)):
            if j == i:
                cur_nfps.append([]), cur_nfps_edges.append([]), cur_nfps_bounds.append([0,0,0,0]), cur_convex_status.append([])
                continue
            nfp = self.getNFP(i, j, oi, self.orientation[j])
            cur_nfps.append(nfp) # 添加NFP
            cur_nfps_edges.append(GeometryAssistant.getPolyEdges(nfp)) # 添加NFP
            cur_nfps_bounds.append(self.getBoundsbyRow(i, j, oi, self.orientation[j], nfp[0])) # 添加边界
            cur_convex_status.append(self.nfps_convex_status[self.computeRow(i, j, oi, self.orientation[j])]) # 添加凹凸情况
            feasible_IFR = feasible_IFR.difference(Polygon(nfp)) # 求解可行区域

        '''如果剩余的面积大于Bias则选择一个点，该阶段不需要考虑在边界的情况'''
        if feasible_IFR.area > self.bias:
            potential_points = GeometryAssistant.kwtGroupToArray(feasible_IFR,0)
            random_index = random.randint(0, len(potential_points) - 1)
            return 0,potential_points[random_index],[0 for _ in range(self.polys_num)]

        '''计算各个阶段的NFP情况'''
        all_search_targets = self.getSearchTargets(i, cur_nfps, cur_nfps_bounds, cur_nfps_edges, ifr_edges, ifr_bounds, ifr) # 获得邻接NFP和交集
        min_pd, best_pt, best_pd_record = 99999999999, [], [0 for _ in range(self.polys_num)]
        for k, search_target in enumerate(all_search_targets):
            pt = [search_target[0],search_target[1]]
            total_pd, pd_record = 0, [0 for _ in range(self.polys_num)]
            for j in search_target[3]:
                if GeometryAssistant.boundsContain(cur_nfps_bounds[j], pt) == False:
                    continue
                if Polygon(cur_nfps[j]).contains(Point(pt)) == True:
                    pd = self.getPolyPtPD(pt,cur_nfps[j],cur_convex_status[j])
                    total_pd, pd_record[j] = total_pd + pd * self.miu[i][j], pd
                    # print(j, "-" , pd)
            if total_pd < min_pd:
                min_pd, best_pt, best_pd_record = total_pd, copy.deepcopy(pt), copy.deepcopy(pd_record)

        return min_pd, best_pt, best_pd_record

    def getSearchTargets(self, index, cur_nfps, cur_nfps_bounds, cur_nfps_edges, ifr_edges, ifr_bounds, ifr):
        '''根据NFP的情况求解交点，并求解最佳位置'''
        nfp_neighbor = [[w] for w in range(len(cur_nfps))] # 全部的点及其对应区域，NFP的邻接多边形
        all_search_targets = [] # [[pt[0],pt[1],[contain_nfp]],[]]

        # 计算全部NFP交点
        for i in range(len(cur_nfps)-1):
            for j in range(i+1, len(cur_nfps)):
                if i == index or j == index:
                    continue
                bounds_i, bounds_j = cur_nfps_bounds[i], cur_nfps_bounds[j] # 获得边界
                if bounds_i[2] <= bounds_j[0] or bounds_i[0] >= bounds_j[2] or bounds_i[3] <= bounds_j[1] or bounds_i[1] >= bounds_j[3]:
                    continue
                inter_points, intersects = GeometryAssistant.interBetweenNFPs(cur_nfps_edges[i], cur_nfps_edges[j],ifr_bounds) # 计算NFP之间的交点
                if intersects == True:
                    nfp_neighbor[i].append(j) # 记录邻接情况    
                    nfp_neighbor[j].append(i) # 记录邻接情况
                if len(inter_points) == 0:
                    continue
                for pt in inter_points:
                    all_search_targets.append([pt[0],pt[1],[i,j]]) # 计算直接取0
        
        # 计算全部NFP的顶点
        for i,nfp in enumerate(cur_nfps):
            new_pts = GeometryAssistant.interNFPIFR(nfp, ifr_bounds, ifr_edges)
            all_search_targets = all_search_targets + [[pt[0],pt[1],[i]] for pt in new_pts]
        
        # 根据第一二个位置排序，合并目标多边形
        all_search_targets = sorted(all_search_targets, key = operator.itemgetter(0, 1))
        new_all_search_targets = [] 
        for k,target in enumerate(all_search_targets):
            if abs(target[0] - all_search_targets[k-1][0]) < compute_bias and abs(target[1] - all_search_targets[k-1][1]) < compute_bias:
                new_all_search_targets[-1][2] = list(set(new_all_search_targets[-1][2] + target[2]))
                continue
            new_all_search_targets.append(target)

        # 求解邻域部分（除去边界点）
        for i,search_target in enumerate(new_all_search_targets):
            neighbor = nfp_neighbor[search_target[2][0]]
            for possible_orignal in search_target[2][1:]:
                # neighbor = list(set(neighbor + nfp_neighbor[possible_orignal]))
                neighbor = [k for k in neighbor if k in nfp_neighbor[possible_orignal]]
            simple_neighbor = [k for k in neighbor if k not in search_target[2]]
            new_all_search_targets[i].append(simple_neighbor)

        return new_all_search_targets

    def updatePD(self, choose_index, final_pd_record):
        '''平移某个元素后更新对应的PD'''
        for i in range(len(self.polys)):
            self.pair_pd_record[i][choose_index],self.pair_pd_record[choose_index][i] = final_pd_record[i], final_pd_record[i]

    def getPDStatus(self):
        '''获得当前的最佳情况'''
        total_pd,max_pair_pd = 0,0
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                total_pd = total_pd + self.pair_pd_record[i][j]
                if self.pair_pd_record[i][j] > max_pair_pd:
                    max_pair_pd = self.pair_pd_record[i][j]
        return total_pd,max_pair_pd

    def initialPairPD(self):
        '''获得当前全部形状间的PD，无需调整'''
        self.pair_pd_record = [[0]*len(self.polys) for _ in range(len(self.polys))] # 两两之间的pd和总的pd
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                top_pt, pd = GeometryAssistant.getTopPoint(self.polys[i]), 0
                nfp = self.getNFP(i, j, self.orientation[i], self.orientation[j])
                bounds = self.getBoundsbyRow(i, j, self.orientation[i], self.orientation[j], nfp[0])
                if top_pt[0] <= bounds[0] or top_pt[0] >= bounds[2] or top_pt[1] <= bounds[1] or top_pt[1] >= bounds[3]:
                    continue
                if Polygon(nfp).contains(Point(top_pt)) == True:
                    pd = self.getPolyPtPD(top_pt, nfp, self.nfps_convex_status[self.computeRow(i, j, self.orientation[i], self.orientation[j])])
                self.pair_pd_record[i][j], self.pair_pd_record[j][i] = pd, pd # 更新对应的pd

    def getPolyPtPD(self, pt, nfp, convex_status):
        '''根据顶点和当前NFP的位置与convex status求解PD'''
        min_pd, final_foot_pt = 999999999999, []
        edges = GeometryAssistant.getPolyEdges(nfp)
        for edge in edges:
            foot_pt = GeometryAssistant.getFootPoint(pt,edge[0],edge[1]) # 求解垂足
            if foot_pt[0] < min(edge[0][0],edge[1][0]) or foot_pt[0] > max(edge[0][0],edge[1][0]) or foot_pt[1] < min(edge[0][1],edge[1][1]) or foot_pt[1] > max(edge[0][1],edge[1][1]):
                continue
            pd = math.sqrt(pow(foot_pt[0]-pt[0],2) + pow(foot_pt[1]-pt[1],2))
            if pd < min_pd:
                min_pd, final_foot_pt = pd, copy.deepcopy(foot_pt)
                if min_pd < self.bias:
                    return 0

        for k,nfp_pt in enumerate(nfp):
            if convex_status[k] == 0:
                non_convex_pd = abs(pt[0]-nfp_pt[0]) + abs(pt[1]-nfp_pt[1])
                if non_convex_pd < min_pd:
                    min_pd = non_convex_pd
                    if min_pd < self.bias:
                        return 0
        return min_pd

    def getIndexPD(self,i,top_pt,oi):
        '''获得某个形状的全部PD，是调整后的结果'''
        total_pd = 0 # 获得全部的NFP基础
        for j in range(len(self.polys)):
            total_pd = total_pd + self.pair_pd_record[i][j]*self.miu[i][j] # 计算PD，比较简单
        return total_pd

    def getNFP(self, i, j, oi, oj):
        '''根据形状和角度获得NFP的情况'''
        row = self.computeRow(i, j, oi, oj)
        bottom_pt = GeometryAssistant.getBottomPoint(self.polys[j])
        original_nfp = copy.deepcopy(self.all_nfps[row])
        nfp = GeometryAssistant.getSlide(original_nfp, bottom_pt[0], bottom_pt[1])
        return nfp

    def getBoundsbyRow(self, i, j, oi, oj, first_pt):
        row = self.computeRow(i, j, oi, oj)
        bounds = self.nfps_bounds[row]
        return [bounds[0]+first_pt[0],bounds[1]+first_pt[1],bounds[2]+first_pt[0],bounds[3]+first_pt[1]]

    def computeRow(self, i, j, oi, oj):
        return self.polys_type[j]*self.types_num*len(self.allowed_rotation)*len(self.allowed_rotation) + self.polys_type[i]*len(self.allowed_rotation)*len(self.allowed_rotation) + oj*len(self.allowed_rotation) + oi # i为移动形状，j为固定位置

    def updateMiu(self,max_pair_pd):
        """寻找到更优位置之后更新"""
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                self.miu[i][j] = self.miu[i][j] + self.pair_pd_record[i][j]/max_pair_pd
                self.miu[j][i] = self.miu[j][i] + self.pair_pd_record[j][i]/max_pair_pd
        # print("miu:",self.miu)

    def shrinkBorder(self):
        '''收缩边界，将超出的多边形左移'''
        # 收缩边界宽度
        self.cur_length = self.best_length*(1 - self.ration_dec)
        # 如果超过了100%就定位100%
        if self.total_area/(self.cur_length*self.width) > 1:
            self.cur_length = self.total_area/self.width
        # 把形状全部内移
        for index,poly in enumerate(self.polys):
            right_pt = GeometryAssistant.getRightPoint(poly)
            if right_pt[0] > self.cur_length:
                delta_x = self.cur_length-right_pt[0]
                GeometryAssistant.slidePoly(poly,delta_x,0)
        _str = "当前目标利用率" + str(self.total_area/(self.cur_length*self.width))
        OutputFunc.outputWarning(self.set_name,_str)
    
    def extendBorder(self):
        '''扩大边界'''
        self.cur_length = self.best_length*(1 + self.ration_inc)

    def getPolygon(self, index, orientation):
        '''获得某个形状'''
        return copy.deepcopy(self.all_polygons[self.polys_type[index]][orientation])

    def initialProblem(self, index):
        '''获得某个解，基于该解进行优化'''
        _input = pd.read_csv("record/lp_initial.csv")
        self.set_name = _input["set_name"][index]
        self.width = _input["width"][index]
        self.bias = _input["bias"][index]
        self.max_overlap = _input["max_overlap"][index]
        self.allowed_rotation = json.loads(_input["allowed_rotation"][index])
        self.total_area = _input["total_area"][index]
        self.polys, self.best_polys = json.loads(_input["polys"][index]), json.loads(_input["polys"][index]) # 获得形状
        self.polys_type = json.loads(_input["polys_type"][index]) # 记录全部形状的种类
        self.polys_num = len(self.polys)
        self.types_num = _input["types_num"][index] # 记录全部形状的种类
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
        fu = pd.read_csv("data/" + self.set_name + "_orientation.csv") 
        # 加载全部形状
        for i in range(fu.shape[0]):
            polygons=[]
            for j in self.allowed_rotation:
                polygons.append(json.loads(fu["o_"+str(j)][i]))
            self.all_polygons.append(polygons)
        # 加载是否对称
        if len(self.allowed_rotation) >= 2:
            self.vertical_symmetrical = []
            for i in range(fu.shape[0]):
                self.vertical_symmetrical.append(fu["ver_sym"][i])
        if len(self.allowed_rotation) == 4:
            self.horizon_symmetrical = []
            for i in range(fu.shape[0]):
                self.horizon_symmetrical.append(fu["hori_sym"][i])
        # 加载NFP的属性
        self.nfps_convex_status,self.nfps_vertical_direction,self.nfps_bounds,self.all_nfps = [],[],[],[]
        nfps = pd.read_csv("data/" + self.set_name + "_nfp.csv") # 获得NFP
        for i in range(nfps.shape[0]):
            self.nfps_convex_status.append(json.loads(nfps["convex_status"][i]))
            self.nfps_vertical_direction.append(json.loads(nfps["vertical_direction"][i]))
            self.nfps_bounds.append(json.loads(nfps["bounds"][i]))
            self.all_nfps.append(json.loads(nfps["nfp"][i]))

    def showPolys(self,saving = False, coloring = None):
        '''展示全部形状以及边框'''
        for poly in self.polys:
            if coloring != None and poly == coloring:
                PltFunc.addPolygonColor(poly,"red") # 红色突出显示
            else:
                PltFunc.addPolygon(poly)
        PltFunc.addPolygonColor([[0,0], [self.cur_length,0], [self.cur_length,self.width], [0,self.width]])
        PltFunc.showPlt(width=1500, height=1500)
        # PltFunc.showPlt(width=2500, height=2500)

if __name__=='__main__':
    GSMPD()
    # for i in range(100):
    #     permutation = np.arange(10)
    #     np.random.shuffle(permutation)
    #     print(permutation)
