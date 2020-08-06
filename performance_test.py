"""
通过封装函数进行性能测试
Created on Thu Aug 6, 2020
"""
from tools.geo_assistant import GeometryAssistant
from random import random, choice
from json import loads
from pandas import read_csv
from time import time
import matplotlib.pyplot as plt

class PD_test(object):
    def __init__(self, index, precision):
        '''
        precision: [grid_precision, digital_precision]
        '''
        self.TEST_NUM = 100000
        self.precision = precision
        self.threshold = precision[0]*0.75
        self.bias, self.compute_bias = 0.1, 1e-7
        self.initialProblem(index)

    def main(self):
        self.stat = [0, 0, 0, 0, 0, 0, 0]
        start = time()
        for i in range(self.TEST_NUM):
            i,j = choice(self.polys_type), choice(self.polys_type)
            oi,oj = choice(self.orientation), choice(self.orientation)
            row = self.computeRow(i,j,oi,oj)
            bounds = self.nfps_bounds[row]
            pt = [bounds[0]+(bounds[2]-bounds[0])*random(), bounds[1]+(bounds[3]-bounds[1])*random()]
            self.getPolyPtPD(pt, row)
        end = time()
        print('精确度:{}, 运行时间: {}'.format(self.precision, end-start))
        print('存在grid, 使用grid, 存在digital, 存在exterior, 算出不包含, 总计算, 冗余计算: {}'.format(self.stat))
        file = open('record/pd_test.csv', 'a')
        res = str(self.precision[0])+', '+str(end-start)+', '+str(self.stat).replace('[', '').replace(']', '')
        file.write(res+'\n')
        file.close()
        return end-start

    def getPolyPtPD(self, pt, row):
        '''Step 1 首先处理参考点和全部（已经判断了是否包含Bounds）'''
        grid_precision, digital_precision = self.precision[0], self.precision[1]
        nfp, nfp_parts, convex_status = self.all_nfps[row], self.nfp_parts[row], self.nfps_convex_status[row]
        relative_pt = [pt[0] - nfp[0][0], pt[1] - nfp[0][1]]
        grid_pt, grid_key = self.getAdjustPt(relative_pt, grid_precision, 5)
        digital_pt, digital_key = self.getAdjustPt(relative_pt, digital_precision, 6)
        original_grid_pt, original_digital_pt = [grid_pt[0]+nfp[0][0], grid_pt[1]+nfp[0][1]], [digital_pt[0]+nfp[0][0], digital_pt[1]+nfp[0][1]]

        '''Step 2 判断是否存在于last_grid_pds和last_digital_pds'''
        if grid_key in self.last_grid_pds[row]:
            self.stat[0] = self.stat[0]+1
            possible_pd = self.last_grid_pds[row][grid_key]
            if possible_pd >= self.threshold: # 如果比较大则直接取该值
                self.stat[1] = self.stat[1]+1
                return possible_pd
            if digital_key in self.last_digital_pds[row]: # 如果存在具体的位置
                self.stat[2] = self.stat[2]+1
                return self.last_digital_pds[row][digital_key]
            
        '''Step 3 判断是否在外部和内外部情况'''
        if digital_key in self.last_exterior_pts[row]:
            self.stat[3] = self.stat[3]+1
            return 0

        if len(nfp_parts) > 0:
            if not GeometryAssistant.judgeContain(relative_pt,nfp_parts):
                self.last_exterior_pts[row][digital_key] = 1
                self.stat[4] = self.stat[4]+1
                return 0
        else:
            print('nfp_parts error')

        '''Step 4 求解PD结果（存在冗余计算）'''
        grid_pd = GeometryAssistant.getPtNFPPD(original_grid_pt, convex_status, nfp, self.bias)
        self.last_grid_pds[row][grid_key] = grid_pd
        self.stat[5] = self.stat[5]+1
        if grid_pd < self.threshold:
            if abs(digital_pt[0]-grid_pt[0])<self.compute_bias and abs(digital_pt[1]-grid_pt[1])<self.compute_bias:
                digital_pd = grid_pd
            else:
                self.stat[6] = self.stat[6]+1
                digital_pd = GeometryAssistant.getPtNFPPD(original_digital_pt, convex_status, nfp, self.bias)
            self.last_digital_pds[row][digital_key] = digital_pd
            return digital_pd

        return grid_pd

    def getAdjustPt(self, pt, precision, zfill_num):
        new_pt = [int(pt[0]/precision+0.5)*precision, int(pt[1]/precision+0.5)*precision]
        target_key = str(int(new_pt[0]/precision)).zfill(zfill_num) + str(int(new_pt[1]/precision)).zfill(zfill_num)
        return new_pt, target_key

    def computeRow(self, i, j, oi, oj):
        return self.polys_type[j]*self.types_num*len(self.allowed_rotation)*len(self.allowed_rotation) + self.polys_type[i]*len(self.allowed_rotation)*len(self.allowed_rotation) + oj*len(self.allowed_rotation) + oi # i为移动形状，j为固定位置

    def initialProblem(self, index):
        '''获得某个解，基于该解进行优化'''
        _input = read_csv("record/lp_initial.csv")
        self.set_name = _input["set_name"][index]
        self.width = _input["width"][index]
        self.bias = _input["bias"][index]
        self.max_overlap = _input["max_overlap"][index]
        self.allowed_rotation = loads(_input["allowed_rotation"][index])
        self.total_area = _input["total_area"][index]
        self.polys, self.best_polys = loads(_input["polys"][index]), loads(_input["polys"][index]) # 获得形状
        self.polys_type = loads(_input["polys_type"][index]) # 记录全部形状的种类
        self.polys_num = len(self.polys)
        self.types_num = _input["types_num"][index] # 记录全部形状的种类
        self.orientation, self.best_orientation = loads(_input["orientation"][index]),loads(_input["orientation"][index]) # 当前的形状状态（主要是角度）
        self.total_area = _input["total_area"][index] # 用来计算利用率
        self.use_ratio = [] # 记录利用率
        self.getPreData() # 获得NFP和全部形状
        self.cur_length = GeometryAssistant.getPolysRight(self.polys) # 获得当前高度
        self.best_length = self.cur_length # 最佳高度
        print("一共",self.polys_num,"个形状")
        self.last_grid_pds = [{} for row in range(len(self.all_nfps))]
        self.last_digital_pds = [{} for row in range(len(self.all_nfps))]
        self.last_exterior_pts = [{} for row in range(len(self.all_nfps))]

    def getPreData(self):
        '''获得全部的NFP和各个方向的形状'''
        self.all_polygons = [] # 存储所有的形状及其方向
        fu = read_csv("data/" + self.set_name + "_orientation.csv") 
        # 加载全部形状
        for i in range(fu.shape[0]):
            polygons=[]
            for j in self.allowed_rotation:
                polygons.append(loads(fu["o_"+str(j)][i]))
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
        self.nfps_convex_status,self.nfps_vertical_direction,self.nfps_bounds,self.all_nfps,self.nfp_parts = [],[],[],[],[]
        nfps = read_csv("data/" + self.set_name + "_nfp.csv") # 获得NFP
        for i in range(nfps.shape[0]):
            self.nfps_convex_status.append(loads(nfps["convex_status"][i]))
            self.nfps_vertical_direction.append(loads(nfps["vertical_direction"][i]))
            self.nfps_bounds.append(loads(nfps["bounds"][i]))
            self.all_nfps.append(loads(nfps["nfp"][i]))
            self.nfp_parts.append(loads(nfps["nfp_parts"][i]))

t=[]
for p in range(1,200):
    t.append(PD_test(2,[p,1]).main())

plt.plot(range(1,200),t)
plt.show()