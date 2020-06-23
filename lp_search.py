"""
Hybrid Algorithm：Guided LP Search+Separation
-----------------------------------
Created on Wed Dec 11, 2020
@author: seanys,prinway
-----------------------------------
"""
from tools.polygon import GeoFunc,PltFunc,getData,getConvex,NFP
from tools.packing import PolyListProcessor,NFPAssistant,PackingUtil
from tools.lp_assistant import LPAssistant
from compaction import searchForBest,LPFunction
from shapely.geometry import Polygon,Point,mapping,LineString
from interval import Interval
import pandas as pd
import json
import copy
import random
import math
import datetime
import time
import csv # 写csv
import numpy as np
import matplotlib.pyplot as plt

bias=0.0000001

class LPSearch(object):
    '''
    线性检索算法，采用数据集Fu
    '''
    def __init__(self,width,original_polys):
        self.width=width
        self.polys=copy.deepcopy(original_polys)
        self.fu=pd.read_csv("/Users/sean/Documents/Projects/Data/fu_orientation.csv")
        self.fu_pre=pd.read_csv("/Users/sean/Documents/Projects/Data/fu.csv")
        self.NFPAssistant=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)

        self.getAllPolygons()
        self.getInitialResult()
        self.main()

    # 获得初始解
    def getInitialResult(self):
        index=6
        blf = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/blf.csv")
        self.total_area=blf["total_area"][index]
        self.polys=json.loads(blf["polys"][index])
        self.best_polys= copy.deepcopy(self.polys)# 按照index的顺序排列
        self.best_poly_status,self.poly_status=json.loads(blf["poly_status"][index]),json.loads(blf["poly_status"][index])
        self.use_ratio=[]
        # 在没有的时候全部加载一遍
        if len(self.best_poly_status)==0:
            for i,poly in enumerate(self.polys):
                top_pt=LPAssistant.getTopPoint(poly)
                self.best_poly_status.append([i,top_pt,0]) # 分别为序列号、位置及方向
                self.poly_status.append([i,top_pt,0]) # 分别为序列号、位置及方向
        print("一共",len(self.polys),"个形状")
    
    # 主要执行过程
    def main(self):
        ration_dec,ration_inc=0.04,0.01
        max_time=2000
        print("执行主程序")
        self.best_length=LPAssistant.getLength(self.polys) # 最佳状态
        print("初始高度:",self.best_length)
        self.cur_length=self.best_length*(1-ration_dec) # 当前的宽度

        self.slideToContainer() # 把突出去的移进来

        start_time = time.time()
        self.use_ratio.append(self.total_area/(self.best_length*self.width))
        print("当前利用率：",self.total_area/(self.best_length*self.width))

        while time.time()-start_time<max_time:
            # 最小化重叠
            self.minimizeOverlap()
            if LPAssistant.judgeFeasible(self.polys)==True:
                # 更新全部状态
                self.length=self.cur_length
                self.use_ratio.append(self.total_area/(self.length*self.width))
                print("当前利用率：",self.total_area/(self.length*self.width))
                self.best_polys=copy.deepcopy(self.polys)
                self.best_poly_status=copy.deepcopy(self.poly_status)
                # 收缩边界，并且把突出去的移进来
                self.cur_length=self.length*(1-ration_dec)
                self.slideToContainer() 
            else:
                # 如果不可行就直接拆分
                self.cur_length=self.best_length*(1+ration_inc)

                
      
        end_time = time.time()
        print("最优结果：",self.best_polys)
        self.showPolys()
        self.plotRecord("use ratio:",self.use_ratio)

    # 最小化重叠区域
    def minimizeOverlap(self):
        start_time=time.time()

        # 记录引导检索的相关内容
        self.miu=[[1]*len(self.polys) for _ in range(len(self.polys))] 
        self.initialOverlap()

        # 记录重叠变化情况
        self.overlap_reocrd=[]

        # 检索次数限制/超出倍数退出
        it,N=0,50
        minimal_overlap=self.getTotalOverlap()
        cur_overlap=minimal_overlap
        print("初始重叠:",cur_overlap)

        # 限定计算次数
        print("开始一次检索")
        while it<N:
            print("it:",it)
            # 获得随机序列并逐一检索
            permutation = np.arange(len(self.polys))
            np.random.shuffle(permutation)
            for i in range(len(self.polys)):
                # 选择特定形状
                choose_index=permutation[i]

                # 通过重叠判断是否需要计算
                with_overlap=False
                for item in self.pair_overlap[choose_index]:
                    if item>0:
                        with_overlap=True
                        break
                if with_overlap==False:
                    continue

                # 获得当前的最小的深度（调整后），如果没有重叠，直接下一个
                self.getPrerequisite(choose_index,self.poly_status[choose_index][2],offline=True)
                cur_min_depth=self.getPolyDepeth(choose_index)

                # 记录最优情况，默认是当前情况
                original_position=self.poly_status[choose_index][1]
                best_position,best_orientation,best_depth=self.poly_status[choose_index][1],self.poly_status[choose_index][2],cur_min_depth
                # print("当前最低高度:",best_depth)

                print("测试第",i,"个形状")
                # 遍历四个角度的最优值
                for orientation in [0,1,2,3]:
                    # print("测试角度:",90*orientation,"度")
                    self.getPrerequisite(choose_index,orientation,offline=True)
                    self.getProblemLP()
                    new_position,new_depth=self.searchBestPosition(choose_index) # 获得最优位置
                    if new_depth<best_depth:
                        best_position,best_orientation,best_depth=copy.deepcopy(new_position),orientation,new_depth

                # 如果有变化状态则需要更新overlap以及移动形状
                if best_position!=original_position:
                    print("本次检索最低深度：",best_depth)
                    # 更新记录的位置
                    self.poly_status[choose_index][1]=copy.deepcopy(best_position)
                    self.poly_status[choose_index][2]=best_orientation
                    # 获取形状顶部位置并平移过去
                    new_poly=copy.deepcopy(self.all_polygons[choose_index][best_orientation])
                    top_point=LPAssistant.getTopPoint(new_poly)
                    GeoFunc.slidePoly(new_poly,best_position[0]-top_point[0],best_position[1]-top_point[1])
                    # 更新形状与重叠情况
                    self.polys[choose_index]=new_poly
                    self.updateOverlap(choose_index)
                    # self.showPolys()

            # 计算新方案的重叠情况
            cur_overlap=self.getTotalOverlap()
            self.overlap_reocrd.append(cur_overlap)
            if cur_overlap<bias:
                print("没有重叠，本次检索结束")
                break
            elif cur_overlap<minimal_overlap:
                minimal_overlap=cur_overlap
                it=0
            print("\n当前重叠:",cur_overlap,"\n")
            it=it+1
            self.updateMiu()
        
        # 超出检索次数
        if it==N:
            print("超出更新次数/超出倍数")
            # self.showPolys()

        end_time=time.time()
        print("本轮耗时：",end_time-start_time)
        print("最终结果：",self.polys)
        print("当前状态：",self.poly_status)

        with open("/Users/sean/Documents/Projects/Packing-Algorithm/record/fu_result.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([[time.asctime( time.localtime(time.time()) ),end_time-start_time,self.cur_length,self.total_area/(self.cur_length*self.width),cur_overlap,self.poly_status,self.polys]])
        
        self.showPolys()
        self.plotRecord("Overlap Record:",self.overlap_reocrd)

    # 获得全部形状不同方向-存储起来
    def getAllPolygons(self):
        self.all_polygons=[]
        for i in range(self.fu.shape[0]):
            polygons=[]
            for j in ["o_0","o_1","o_2","o_3"]:
                polygons.append(json.loads(self.fu[j][i]))
            self.all_polygons.append(polygons)
        
    # 获得整个重叠情况
    def getTotalOverlap(self):
        # print(self.pair_overlap)
        overlap=0
        for i in range(len(self.pair_overlap)-1):
            for j in range(i+1,len(self.pair_overlap[0])):
                overlap=overlap+self.pair_overlap[i][j]
        return overlap

    # 初始化全部的重叠情况
    def initialOverlap(self):
        self.pair_overlap=[[0]*len(self.polys) for i in range(len(self.polys))]
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                P1,P2=Polygon(self.polys[i]),Polygon(self.polys[j])
                inter=P1.intersection(P2) # 获得重叠区域
                if inter.area>bias:
                    self.pair_overlap[i][j]=self.pair_overlap[i][j]+inter.area
                    self.pair_overlap[j][i]=self.pair_overlap[j][i]+inter.area

    # 更新目标对象的Overlap
    def updateOverlap(self,choose_index):
        # 重新计算该对象的全部重叠
        for j in range(len(self.polys)):
            if j == choose_index:
                continue
            P1,P2=Polygon(self.polys[choose_index]),Polygon(self.polys[j])
            inter=P1.intersection(P2) # 获得重叠区域
            inter_area=0
            if inter.area>bias:
                inter_area=inter.area        
            self.pair_overlap[choose_index][j]=inter_area
            self.pair_overlap[j][choose_index]=inter_area
    
    # 基于NFP获得全部的约束
    def getProblemLP(self):
        # 获得目标区域
        self.ifr_points=[]
        self.target_areas=[[],[],[],[],[],[],[],[],[]]
        self.last_index=[[],[],[],[],[],[],[],[],[]]
        
        # 获得两个NFP在IFR中重叠的情况
        self.nfp_overlap_pair=[[i] for i in range(len(self.all_nfps))]
        for i in range(len(self.all_nfps)-1):
            for j in range(i+1,len(self.all_nfps)):
                overlap,overlap_poly=self.polysOverlapIFR(self.all_nfps[i],self.all_nfps[j])
                if overlap==True:
                    self.nfp_overlap_pair[i].append(j)
                    self.nfp_overlap_pair[j].append(i)
                    self.target_areas[1].append([overlap_poly,i,j])
                    self.last_index[1].append([i,j]) # 分别添加i,j
        
        # 切去一维重叠情况
        for i,nfp in enumerate(self.all_nfps):
            # 删除与IFR重叠区域
            new_region=Polygon(nfp).intersection(self.IFR)
            self.final_IFR=self.final_IFR.difference(Polygon(nfp))
            # 删除与其他NFP拆分的重叠
            for j in self.nfp_overlap_pair[i][1:]:
                P=Polygon(self.all_nfps[j])
                new_region=new_region.difference(P)
            # 在目标区域增加情况，首先排除点和直线，以及面积过小
            if new_region.is_empty!=True and new_region.geom_type!="Point" and new_region.geom_type!="LineString" and new_region.area>bias:
                self.target_areas[0].append([LPAssistant.processRegion(new_region),i]) # 删除直线/顶点情况
            else:
                self.target_areas[0].append([])
            self.last_index[0].append([])
        
        # 增加IFR的计算
        if self.final_IFR.is_empty!=True and self.final_IFR.geom_type!="Point" and self.final_IFR.geom_type!="LineString" and self.final_IFR.area>bias:
            self.ifr_points=LPAssistant.processRegion(self.final_IFR)
        
        # 获得后续的重叠
        for i in range(2,len(self.target_areas)):
            # 遍历上一阶段计算的结果
            for j,target_area in enumerate(self.target_areas[i-1]):
                area,P1=target_area[0],Polygon(target_area[0]) # 获得当前目标可行解
                
                all_possible_target=[]
                # 如果大于三个，只需要计算最后加入的，否则是第一个
                if i>=3:
                    all_possible_target=self.nfp_overlap_pair[target_area[-1]]
                else:
                    all_possible_target=self.nfp_overlap_pair[target_area[1]]+self.nfp_overlap_pair[target_area[2]]
            
                all_possible_target=PolyListProcessor.deleteRedundancy(all_possible_target)

                # 删除所有更小的，保证正序，获得判断这些形状是否会重叠，若有则添加并求解目标区域
                all_possible_target_larger=LPAssistant.deleteTarget(all_possible_target,[i for i in range(0,max(item for item in target_area[1:])+1)])
                for possible_target in all_possible_target_larger:
                    P2=Polygon(self.all_nfps[possible_target])
                    # 只有相交才进一步计算
                    if P1.intersects(P2):
                        inter=P1.intersection(P2)
                        if inter.area>bias:
                            self.last_index[i].append([j])
                            self.target_areas[i].append([LPAssistant.processRegion(inter)]+target_area[1:]+[possible_target])
                
                # 删除已经有的，遍历计算重叠
                all_possible_target_difference=LPAssistant.deleteTarget(all_possible_target,[item for item in target_area[1:]])
                new_region=self.cutFrontRegion(all_possible_target_difference,P1)
                
                if new_region.is_empty!=True and new_region.geom_type!="Point" and new_region.geom_type!="LineString" and new_region.area>bias:
                    target_area[0]=LPAssistant.processRegion(new_region)
                else:
                    self.target_areas[i-1][j]=[]

            # 如果该轮没有计算出重叠则停止
            if self.target_areas[i]==[]:
                self.max_overlap=i
                # print("至多",i,"个形状重叠，计算完成")
                break

    # 删除重复情况
    def cutFrontRegion(self,all_possible_target_difference,P1):
        '''根据可行区域计算切除的结果'''
        new_region=copy.deepcopy(P1)
        for difference_target in all_possible_target_difference:
            P2=Polygon(self.all_nfps[difference_target])
            if new_region.intersects(P2):
                new_region=new_region.difference(P2)
        return new_region

    def searchBestPosition(self,choose_index):
        '''基于上述获得的区域与目标函数检索最优位置'''
        min_depth,best_position,searched_points=9999999999,[],[]
        # 首先判断在IFR上是否有点满足
        if len(self.ifr_points)>0:
            return self.ifr_points[random.randint(0,len(self.ifr_points)-1)],0

        # 再选择是否有目标区域
        for i,item in enumerate(self.target_areas):
            for j,area_item in enumerate(item):
                # 计算差集后归零
                if len(area_item)==0:
                    continue
                # 分别计算每个点在每个区域的最值
                for pt in area_item[0]:
                    # 防止重复计算问题
                    if pt in searched_points:
                        continue
                    searched_points.append(pt)
                    # 计算全部重叠
                    depth=0
                    for target_index in area_item[1:]:
                        depth=depth+self.getPairDepenetration(pt,choose_index,target_index)
                    if depth<min_depth:
                        min_depth=depth
                        best_position=[pt[0],pt[1]]
        print("共检索",len(searched_points),"个位置")
        return best_position,min_depth
    
    # 获得当前选择对象的重叠（对应的Overlap）
    def getPolyDepeth(self,index):
        cur_min_depth,pt=0,LPAssistant.getTopPoint(self.polys[index])
        for j in range(len(self.polys)):
            if j==index or self.pair_overlap[index][j]==0:
                continue
            cur_min_depth=cur_min_depth+self.getPairDepenetration(pt,index,j)
        return cur_min_depth

    # 获得当前选择形状和其他形状对的深度（调整后），需要确认二者是重叠的！
    def getPairDepenetration(self,pt,choose_index,target_index):
        min_value=999999999
        for item in self.all_points_target[target_index]:
            value=abs(pt[0]-item[0])+abs(pt[1]-item[1])
            if value<bias:
                min_value=0
                break
            if value<min_value:
                min_value=value
        for item in self.all_edges_target[target_index]:
            value=abs(pt[0]*item[0]+pt[1]*item[1]+item[2])
            if value<bias:
                min_value=0
                break
            if value<min_value:
                min_value=value
        return min_value*self.miu[target_index][choose_index]

    def polysOverlapIFR(self,poly1,poly2):
        '''判断两个形状之间是否重叠、重叠区域面积、重叠区域是否与IFR有重叠'''
        P1,P2=Polygon(poly1),Polygon(poly2)
        inter=P1.intersection(P2)
        overlap,overlap_poly=False,[]
        if inter.area>bias:
            new_inter=inter.intersection(self.IFR)
            if new_inter.area>bias:
                overlap,overlap_poly=True,LPAssistant.processRegion(new_inter) # 相交区域肯定是凸多边形
        return overlap,overlap_poly

    def slideToContainer(self):
        # 平移部分形状
        for index,poly in enumerate(self.polys):
            right_pt=LPAssistant.getRightPoint(poly)
            if right_pt[0]>self.cur_length:
                delta_x=self.cur_length-right_pt[0]
                GeoFunc.slidePoly(poly,delta_x,0)
                top_pt=self.poly_status[index][1]
                self.poly_status[index][1]=[top_pt[0]+delta_x,top_pt[1]]

    def showPolys(self):
        for poly in self.polys:
            PltFunc.addPolygon(poly)
        PltFunc.addPolygonColor([[0,0],[self.cur_length,0],[self.cur_length,self.width],[0,self.width]])
        PltFunc.showPlt(width=1000,height=1000)

    def updateMiu(self):
        # 首先获得Overlap的最大值
        print("更新Miu")
        _max=0
        for row in self.pair_overlap:
            row_max=max(row)
            if row_max>_max:
                _max=row_max
        # 更新Miu的值
        for i in range(len(self.miu)):
            for j in range(i,len(self.miu[0])):
                self.miu[j][i]=self.miu[j][i]+self.pair_overlap[i][j]/_max
                self.miu[i][j]=self.miu[i][j]+self.pair_overlap[i][j]/_max
        # print(self.miu)
    
    def getNFP(self,j,i):
        # j是固定位置，i是移动位置
        row=j*192+i*16+self.poly_status[j][2]*4+self.poly_status[i][2]
        bottom_pt=LPAssistant.getBottomPoint(self.polys[j])
        delta_x,delta_y=bottom_pt[0],bottom_pt[1]
        nfp=GeoFunc.getSlide(json.loads(self.fu_pre["nfp"][row]),delta_x,delta_y)
        return nfp

    # 直接读取目标情况-带方向
    def getPrerequisite(self,i,orientation,**kw):
        # 获得全部NFP以及拆分情况
        self.all_nfps,self.all_points_target,self.all_edges_target = [],[],[]
        offline = kw['offline']
        for j,item in enumerate(self.polys):
            # 两个相等的情况，跳过否则会计算错误
            if j == i:
                self.all_nfps.append([])
                self.all_points_target.append([])
                self.all_edges_target.append([])
                continue
            # 预处理的情况
            points_target,edges_target,nfp = [],[],[]
            if offline == True:
                row = j*192+i*16+self.poly_status[j][2]*4+orientation
                bottom_pt = LPAssistant.getBottomPoint(self.polys[j])
                delta_x,delta_y = bottom_pt[0],bottom_pt[1]
                nfp = GeoFunc.getSlide(json.loads(self.fu_pre["nfp"][row]),delta_x,delta_y)
            else:
                nfp = LPAssistant.deleteOnline(self.NFPAssistant.getDirectNFP(self.polys[j],self.polys[i])) # NFP可能有同一直线上的点
            # 计算对应目标函数
            for pt_index in range(len(nfp)):
                edges_target.append(LPAssistant.getTargetFunction([nfp[pt_index-1],nfp[pt_index]]))
                points_target.append([nfp[pt_index][0],nfp[pt_index][1]])
            # 添加上去
            self.all_nfps.append(nfp)
            self.all_edges_target.append(edges_target)
            self.all_points_target.append(points_target)

        # 获取IFR
        self.target_poly = self.all_polygons[i][orientation]
        self.ifr = PackingUtil.getInnerFitRectangle(self.target_poly,self.cur_length,self.width)
        self.IFR = Polygon(self.ifr)
        self.final_IFR = Polygon(self.ifr)

    @staticmethod
    def plotRecord(name,data):
        plt.plot(data)
        plt.ylabel(name)
        plt.xlabel('Times')
        plt.show()

if __name__=='__main__':
    # polys=getConvex(num=5)
    polys=getData()
    # nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    # print(datetime.datetime.now(),"计算完成NFP")
    # blf=BottomLeftFill(760,polys,vertical=False,NFPAssistant=nfp_ass)
    # print(blf.polygons)
    LPSearch(760,polys)

    # testNonConvex()
    # PltFunc.addPolygon([[460,760],[654.3264153600001,760],[654.3264153600001,280],[460,280],[460,760]])
    # PltFunc.showPlt()