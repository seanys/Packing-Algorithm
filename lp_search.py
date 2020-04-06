'''
2020年3月25日 Use Linear Programming Search New Position
'''
from tools.polygon import GeoFunc,PltFunc,getData,getConvex
from tools.packing import PolyListProcessor,NFPAssistant,PackingUtil
from tools.lp import sovleLP
from heuristic import BottomLeftFill
import pandas as pd
import json
from shapely.geometry import Polygon,Point,mapping,LineString
from interval import Interval
import copy
import random
import math
import datetime
import time
import numpy as np

bias=0.0000001

class LPSearch(object):
    '''
    线性检索算法，采用数据集Fu
    '''
    def __init__(self,width,original_polys):
        self.width=width
        self.polys=copy.deepcopy(original_polys)
        self.fu=pd.read_csv("/Users/sean/Documents/Projects/Data/fu_orientation.csv",header=None)
        self.fu_pre=pd.read_csv("/Users/sean/Documents/Projects/Data/fu.csv")
        # self.NFPAssistant=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)

        self.getAllPolygons()
        self.getInitialResult()
        self.main()

    # 获得初始解
    def getInitialResult(self):
        blf = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/blf.csv")
        self.polys=json.loads(blf["polys"][2])
        self.best_polys= copy.deepcopy(self.polys)# 按照index的顺序排列
        self.best_poly_status,self.poly_status=[],[]
        for i,poly in enumerate(self.polys):
            top_pt=self.getTopPoint(poly)
            self.best_poly_status.append([i,top_pt,0]) # 分别为序列号、位置及方向
            self.poly_status.append([i,top_pt,0]) # 分别为序列号、位置及方向
        print("一共",len(self.polys),"个形状")
    
    # 主要执行过程
    def main(self):
        ration_dec,ration_inc=0.05,0.025
        max_time=0.0001

        self.length=self.getLength() # 最佳状态
        self.cur_length=self.length*(1-ration_dec) # 当前的宽度
        self.slideToContainer() # 把突出去的移进来

        start_time = time.time()
        while time.time()-start_time<max_time:
            self.minimizeOverlap()
            if self.judegeFeasible()==True:
                # 更新全部状态
                self.length=self.getLength()
                self.best_polys=copy.deepcopy(self.polys)
                self.best_poly_status=copy.deepcopy(self.poly_status)
                # 收缩边界，并且把突出去的移进来
                self.cur_length=self.length*(1-ration_dec)
                self.slideToContainer() 
            else:
                # 如果不可行则在上一次的结果基础上增加，再缩减
                self.cur_length=self.length*(1+ration_inc)
        end_time = time.time()
        self.showPolys()
    
    # 最小化重叠区域
    def minimizeOverlap(self):
        self.miu=[[1]*len(self.polys) for _ in range(len(self.polys))] # 用于引导检索，每次都需要更新
        self.updateOverlap() # 更新当前的重叠情况用于计算fitness
        fitness,it,N=self.getFitness(),0,1
        # 限定计算次数
        while it<N:
            # 获得随机序列并逐一检索
            permutation = np.arange(len(self.polys))
            np.random.shuffle(permutation)
            print("获得序列:",permutation)
            for i in range(len(self.polys)):
                # 选择特定形状
                choose_index=permutation[i]
                # 获得当前的最小的深度（调整后），如果没有重叠，直接下一个
                cur_min_depth=self.getPolyDepeth(choose_index)
                if cur_min_depth==0:
                    print("没有重叠")
                    continue
                print("开始寻找最优情况")
                # 记录最优情况，默认是当前情况
                original_position=self.poly_status[choose_index][1]
                best_position,best_orientation,best_depth=self.poly_status[choose_index][1],self.poly_status[choose_index][2],cur_min_depth
                # 遍历四个角度的最优值
                for orientation in [0,1,2,3]:
                    print("测试角度:",90*orientation,"度")
                    self.getPrerequisiteOffLine(choose_index,orientation)
                    self.getProblemLP()
                    new_position,new_depth=self.searchBestPosition() # 获得最优位置

                    top_point=self.getTopPoint(self.all_polygons[choose_index][orientation])
                    new_polygon=GeoFunc.getSlide(self.all_polygons[choose_index][orientation],new_position[0]-top_point[0],new_position[1]-top_point[1])

                    PltFunc.addPolygonColor(new_polygon)
                    PltFunc.addPolygonColor(self.ifr)
                    self.showPolys()

                    if new_depth<best_depth:
                        best_position,best_orientation,best_depth=new_position,orientation,new_depth
                # 如果有变化状态则需要更新overlap以及移动形状
                if best_position!=original_position:
                    # 更新记录的位置
                    self.poly_status[choose_index][1]=best_position
                    self.poly_status[choose_index][2]=best_orientation
                    # 获取形状顶部位置并平移过去
                    new_poly=copy.deepcopy(self.all_polygons[choose_index][best_orientation])
                    top_point=self.getTopPoint(new_poly)
                    GeoFunc.getSlide(new_poly,best_position[0]-top_point[0],best_position[1]-top_point[1])
                    # 更新形状与重叠情况
                    self.polys[choose_index]=new_poly
                    self.updateOverlap()
            # 计算新方案的重叠情况
            cur_fitness=self.getFitness()
            if cur_fitness==0:
                break
            elif cur_fitness<fitness:
                fitness=cur_fitness
                it=0
            # 如果没有更新则会增加（更新了的话会归零）
            it=it+1
            # 更新全部的Miu
            self.updateMiu()
    
    # 获得全部形状不同方向-存储起来
    def getAllPolygons(self):
        self.all_polygons=[]
        for i in range(self.fu.shape[0]):
            polygons=[]
            for j in [0,1,2,3]:
                polygons.append(json.loads(self.fu[j][i]))
            self.all_polygons.append(polygons)
    
    # 获得调整后的某个形状的fitness
    def getPolyDepeth(self,index):
        cur_min_depth=0
        for j in range(len(self.polys)):
            cur_min_depth=cur_min_depth+self.miu[index][j]*self.pair_overlap[index][j]
        return cur_min_depth
    
    # 更新整个的重叠情况
    def updateOverlap(self):
        self.pair_overlap=[[0]*len(self.polys) for i in range(len(self.polys))]
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                P1,P2=Polygon(self.polys[i]),Polygon(self.polys[j])
                inter=P1.intersection(P2) # 获得重叠区域
                if inter.area>bias:
                    self.pair_overlap[i][j]=self.pair_overlap[i][j]+inter.area
                    self.pair_overlap[j][i]=self.pair_overlap[j][i]+inter.area
    
    # 获得整个fitness也是重叠情况
    def getFitness(self):
        fitness=0
        for i in range(len(self.pair_overlap)):
            for j in range(i+1,len(self.pair_overlap[0])):
                fitness=fitness+self.pair_overlap[i][j]
        return fitness

    # 直接读取目标情况-带方向
    def getPrerequisiteOffLine(self,i,orientation):
        # 获得全部NFP以及拆分情况
        self.all_nfps,self.all_divided_nfp,self.all_target_func,self.max_divided_len=[],[],[],0
        for j,item in enumerate(self.poly_status):
            if j == i:
                self.all_nfps.append([])
                self.all_target_func.append([])
                self.all_divided_nfp.append([])
                continue
            
            # 获得初始数据
            row=j*192+i*16+self.poly_status[j][2]*4+orientation
            original_nfp,original_divided_nfp,original_target_func=json.loads(self.fu_pre["nfp"][row]),json.loads(self.fu_pre["divided_nfp"][row]),json.loads(self.fu_pre["target_func"][row])
            # 原点平移到当前位置
            bottom_pt=self.getBottomPoint(self.polys[j])
            delta_x,delta_y=bottom_pt[0],bottom_pt[1]
            # NFP计算结果
            self.all_nfps.append(GeoFunc.getSlide(original_nfp,delta_x,delta_y)) 
            # Divided_nfp处理
            divided_nfp=[]
            for nfp in original_divided_nfp:
                divided_nfp.append(GeoFunc.getSlide(nfp,delta_x,delta_y)) 
            self.all_divided_nfp.append(divided_nfp)
            # 目标函数处理，变化值是(a*delta_x+b*delta_y)*(-1)
            target_func=[]
            for coefficient in original_target_func:
                [a,b,c]=coefficient
                c=c-(a*delta_x+b*delta_y)
                target_func.append([a*self.miu[j][i],b*self.miu[j][i],c*self.miu[j][i]])
            self.all_target_func.append(target_func)

            if len(divided_nfp)>self.max_divided_len:
                self.max_divided_len=len(divided_nfp)

        # 获取IFR
        self.target_poly=self.all_polygons[i][orientation]
        self.ifr=PackingUtil.getInnerFitRectangle(self.target_poly,self.cur_length,self.width)
        self.IFR=Polygon(self.ifr)
    
    @staticmethod
    def getDividedNfp(nfp):
        all_bisectior,divided_nfp,target_func=[],[],[]
        # 遍历NFP的所有顶点计算角平分线
        for i in range(-2,len(nfp)-2):
            vec=LPSearch.getAngularBisector(nfp[i],nfp[i+1],nfp[i+2])
            all_bisectior.append([nfp[i+1],[nfp[i+1][0]+vec[0]*1000,nfp[i+1][1]+vec[1]*1000]])

        # 计算全部的三角形区域和附带边
        divided_nfp,target_func=[],[]
        for i in range(-1,len(all_bisectior)-1):
            line1,line2=all_bisectior[i],all_bisectior[i+1]
            inter=LPSearch.lineIntersection(line1,line2)
            divided_nfp.append([nfp[i-1],nfp[i],inter]) # [边界点1,边界点2,交点]
            target_func.append(LPSearch.getTargetFunction([nfp[i-1],nfp[i]]))

        return all_bisectior,divided_nfp,target_func

    # 基于NFP获得全部的约束
    def getProblemLP(self):
        self.updateNFPOverlap() # 更新NFP的重叠情况
        self.target_areas=[[],[],[],[],[],[],[],[]] # 用于存储目标区域，最高允许8个形状重叠
        self.target_function=[[],[],[],[],[],[],[],[]] # 用于计算目标函数，分别为xy的参数

        '''首先获得全部的一对对的NFP的结果，以及目标函数，然后更新重叠'''
        self.pair_nfps=[[[] for j in range(self.max_divided_len)] for i in range(len(self.polys))]
        self.getSituationTwo()
        self.updateSituationOne()

        '''多个形状的计算'''
        for i in range(2,len(self.target_areas)):
            # 遍历上一阶段计算的结果
            for j,target_area in enumerate(self.target_areas[i-1]):
                area,P1=target_area[0],Polygon(target_area[0]) # 获得当前目标可行解
                
                all_possible_target=self.getAllPossibleTarget(i,target_area) # 其他可能有的区域

                # 删除所有更小的，保证正序，获得判断这些形状是否会重叠，若有则添加并求解目标区域
                all_possible_target_larger=self.deleteTarget(all_possible_target,[i for i in range(0,max(item[0] for item in target_area[1:])+1)])
                self.computeNextOverlap(i,j,target_area,all_possible_target_larger,P1)
                
                # 删除已经有的，遍历计算重叠
                all_possible_target_difference=self.deleteTarget(all_possible_target,[item[0] for item in target_area[1:]])
                new_region=self.cutFrontRegion(all_possible_target_difference,P1)
                
                if new_region.is_empty!=True and new_region.geom_type!="Point" and new_region.geom_type!="LineString" and new_region.area>bias:
                    target_area[0]=self.processRegion(new_region)
                else:
                    self.target_areas[i-1][j]=[]

            # 如果该轮没有计算出重叠则停止
            if self.target_areas[i]==[]:
                print("至多",i,"个形状重叠，计算完成")
                break
    
    # 删除重复情况
    def cutFrontRegion(self,all_possible_target_difference,P1):
        '''根据可行区域计算切除的结果'''
        new_region=copy.deepcopy(P1)
        for difference_target in all_possible_target_difference:
            P2=Polygon(self.all_divided_nfp[difference_target[0]][difference_target[1]])
            if new_region.intersects(P2):
                new_region=new_region.difference(P2)
        return new_region

    # 计算下一个
    def computeNextOverlap(self,i,j,target_area,all_possible_target_larger,P1):
        '''根据可行的情况计算下一个重叠'''
        for possible_target in all_possible_target_larger:
            P2=Polygon(self.all_divided_nfp[possible_target[0]][possible_target[1]])
            # 只有相交才进一步计算
            if P1.intersects(P2):
                inter=P1.intersection(P2)
                if inter.area>bias:
                    self.target_areas[i].append([self.processRegion(inter)]+[item for item in target_area[1:]]+[possible_target])
                    [a1,b1,c1],[a2,b2,c2]=self.target_function[i-1][j],self.all_target_func[possible_target[0]][possible_target[1]]
                    self.target_function[i].append([a1+a2,b1+b2,c1+c2])
    
    # 根据相交情况获得需要计算的区域
    def getAllPossibleTarget(self,i,target_area):
        '''根据i和目标的值获得可能的解'''
        all_possible_target=[]
        if i>=3:
            # 如果是四个形状重叠，只需要增加计算三个形状中最后加入的
            all_possible_target=self.pair_nfps[target_area[-1][0]][target_area[-1][1]]
        else:
            # 需要计算其他两个
            for item in target_area[1:]:
                all_possible_target=all_possible_target+self.pair_nfps[item[0]][item[1]]
            
        all_possible_target=PolyListProcessor.deleteRedundancy(all_possible_target)
        return all_possible_target

    # 获得无重叠情况并删除重复区域
    def updateSituationOne(self):
        '''获得删除重叠区域以及IFR范围的区间'''
        for i,nfp_divided in enumerate(self.all_divided_nfp):
            for j,item in enumerate(nfp_divided):
                # 删除与IFR重叠区域
                new_region=Polygon(item).intersection(self.IFR)
                # 删除与其他NFP拆分的重叠
                for pair in self.pair_nfps[i][j]:
                    P=Polygon(self.all_divided_nfp[pair[0]][pair[1]])
                    new_region=new_region.difference(P)
                # 在目标区域增加情况，首先排除点和直线，以及面积过小
                if new_region.is_empty!=True and new_region.geom_type!="Point" and new_region.geom_type!="LineString" and new_region.area>bias:
                    self.target_areas[0].append([self.processRegion(new_region),[i,j]]) # 删除直线/顶点情况
                    self.target_function[0].append(self.all_target_func[i][j]) 

    # 获得两个形状重叠
    def getSituationTwo(self):
        '''获得二阶可行情况'''
        for i in range(len(self.nfp_overlap_pair)):
            for j in self.nfp_overlap_pair[i][1:]:
                if j<i:
                    continue
                for m,divided_nfp_i in enumerate(self.all_divided_nfp[i]):
                    for n,divided_nfp_j in enumerate(self.all_divided_nfp[j]):
                        overlap,overlap_poly=self.polysOverlapIFR(divided_nfp_i,divided_nfp_j)
                        # 重叠则记录对应的细分重叠
                        if overlap==True:
                            # 记录重叠情况，只记录正序
                            self.pair_nfps[i][m].append([j,n])
                            self.pair_nfps[j][n].append([i,m]) 

                            # 目标区域增加
                            self.target_areas[1].append([overlap_poly,[i,m],[j,n]])
                            # 获得目标参数
                            [a1,b1,c1],[a2,b2,c2]=self.all_target_func[i][m],self.all_target_func[j][n]
                            self.target_function[1].append([a1+a2,b1+b2,c1+c2]) 

    def searchBestPosition(self):
        '''基于上述获得的区域与目标函数检索最优位置'''
        min_depth,best_position=9999999999,[]
        n=0
        for i,item in enumerate(self.target_areas):
            for j,area_item in enumerate(item):
                if len(area_item)==0:
                    continue
                a,b,c=self.target_function[i][j]
                for pt in area_item[0]:
                    n=n+1
                    value=a*pt[0]+b*pt[1]+c
                    if value<min_depth:
                        min_depth=value
                        best_position=[pt[0],pt[1]]
        print("共检索",n,"个位置")
        print("最佳位置：",best_position)
        print("最小重叠：",min_depth)
        return best_position,min_depth

    def judegeFeasible(self):
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                P1,P2=Polygon(self.polys[i]),Polygon(self.polys[j])
                if P1.intersects(P2)==True:
                    return False
        return True

    def polysOverlapIFR(self,poly1,poly2):
        '''判断两个形状之间是否重叠、重叠区域面积、重叠区域是否与IFR有重叠'''
        P1,P2=Polygon(poly1),Polygon(poly2)
        inter=P1.intersection(P2)
        overlap,overlap_poly=False,[]
        if inter.area>bias:
            new_inter=inter.intersection(self.IFR)
            if new_inter.area>bias:
                overlap,overlap_poly=True,GeoFunc.polyToArr(new_inter) # 相交区域肯定是凸多边形
        return overlap,overlap_poly

    def updateNFPOverlap(self):
        '''获得NFP的重叠情况'''
        self.nfp_overlap_pair=[[i] for i in range(len(self.polys))]
        for i in range(len(self.all_nfps)-1):
            for j in range(i+1,len(self.all_nfps)):
                P1,P2=Polygon(self.all_nfps[i]),Polygon(self.all_nfps[j])
                inter=P1.intersection(P2) # 获得重叠区域
                if inter.area>bias:
                    self.nfp_overlap_pair[i].append(j)
                    self.nfp_overlap_pair[j].append(i)
                    
    
    def slideToContainer(self):
        # 平移部分形状
        for index,poly in enumerate(self.polys):
            right_pt=self.getRightPoint(poly)
            if right_pt[0]>self.cur_length:
                delta_x=self.cur_length-right_pt[0]
                GeoFunc.slidePoly(poly,delta_x,0)
                top_pt=self.poly_status[index][1]
                self.poly_status[index][1]=[top_pt[0]+delta_x,top_pt[1]]
    
    def deleteTarget(self,_list,target):
        new_list=[]
        for item in _list:
            existing=False
            for target_item in target:
                if item[0]==target_item:
                    existing=True
            if existing==False:
                new_list.append(item)
        return new_list

    @staticmethod
    def lineIntersection(line1, line2):
        print(line1)
        print(line2)
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return [x, y]

    @staticmethod
    def deleteOnline(poly):
        '''删除两条直线在一个延长线情况'''
        new_poly=[]
        for i in range(-2,len(poly)-2):
            vec1=LPSearch.getDirectionalVector([poly[i+1][0]-poly[i][0],poly[i+1][1]-poly[i][1]])
            vec2=LPSearch.getDirectionalVector([poly[i+2][0]-poly[i+1][0],poly[i+2][1]-poly[i+1][1]])
            if abs(vec1[0]-vec2[0])>bias:
                new_poly.append(poly[i+1])
        return new_poly
    
    @staticmethod
    def matrixDotSum(matrix1,matrix2):
        _sum=0
        for i in len(matrix1):
            for j in len(matrix1[0]):
                _sum=_sum+matrix1[i][j]*matrix2[i][j]
        return _sum

    @staticmethod
    def getDirectionalVector(vec):
        _len=math.sqrt(vec[0]*vec[0]+vec[1]*vec[1])
        return [vec[0]/_len,vec[1]/_len]

    @staticmethod
    def getTargetFunction(edge):
        '''处理NFP拆分的结果，第一条边为边界，只与距离的绝对值有关'''
        A=edge[0][1]-edge[1][1]
        B=edge[1][0]-edge[0][0]
        C=edge[0][0]*edge[1][1]-edge[1][0]*edge[0][1]
        D=math.pow(A*A+B*B,0.5)
        a,b,c=A/D,B/D,C/D
        return [a,b,c]

    # 获得宽度    
    def getLength(self):
        _max=0
        for i in range(0,len(self.polys)):
            extreme_index=GeoFunc.checkRight(self.polys[i])
            extreme=self.polys[i][extreme_index][0]
            if extreme>_max:
                _max=extreme
        return _max
    
    def processRegion(self,region):
        area=[]
        if region.geom_type=="Polygon":
            area=GeoFunc.polyToArr(region)  # 最终结果只和顶点相关
        else:
            for shapely_item in list(region):
                if shapely_item.area>bias:
                    area=area+GeoFunc.polyToArr(shapely_item)
        return area

    @staticmethod
    def getAngularBisector(pt1,pt2,pt3):
        '''
        输入：pt1/pt3为左右两个点，pt2为中间的点
        输出：该角的对角线
        '''
        vec1=LPSearch.getDirectionalVector([pt1[0]-pt2[0],pt1[1]-pt2[1]])
        vec2=LPSearch.getDirectionalVector([pt3[0]-pt2[0],pt3[1]-pt2[1]])
        bisector=[]
        if vec1[0]+vec2[0]==0:
            new_x=math.cos(math.pi/2)*vec1[0] - math.sin(math.pi/2)*vec1[1]
            new_y=math.cos(math.pi/2)*vec1[1] + math.sin(math.pi/2)*vec1[0]
            bisector=[new_x,new_y] # 获得垂直方向，默认选择[pt2,pt3]的右侧
        else:
            bisector=[(vec1[0]+vec2[0]),vec1[1]+vec2[1]] # 获得对角方向，长度为sqrt(2)
        return bisector

    def showPolys(self):
        for poly in self.polys:
            PltFunc.addPolygon(poly)
        PltFunc.addPolygonColor([[0,0],[self.cur_length,0],[self.cur_length,self.width],[0,self.width]])
        PltFunc.showPlt(width=1500,height=1500)

    def getTopPoint(self,poly):
        top_pt,max_y=[],-999999999
        for pt in poly:
            if pt[1]>max_y:
                max_y=pt[1]
                top_pt=[pt[0],pt[1]]
        return top_pt

    def getBottomPoint(self,poly):
        bottom_pt,min_y=[],999999999
        for pt in poly:
            if pt[1]<min_y:
                min_y=pt[1]
                bottom_pt=[pt[0],pt[1]]
        return bottom_pt

    def getRightPoint(self,poly):
        right_pt,max_x=[],-999999999
        for pt in poly:
            if pt[0]>max_x:
                max_x=pt[0]
                right_pt=[pt[0],pt[1]]
        return right_pt

    def updateMiu(self):
        # 首先获得Overlap的最大值
        _max=0
        for row in self.pair_overlap:
            row_max=max(row)
            if row_max>_max:
                _max=row_max
        # 更新Miu的值
        for i in range(len(self.miu)):
            for j in range(len(self.miu[0])):
                self.miu[i][j]=self.miu[i][j]+self.pair_overlap[i][j]/_max

    # 获得全部的NFP及其拆分结果，获得IFR和限制
    def getPrerequisite(self,index):
        '''
        可以和Offline的合并一下
        '''
        # 首先获得NFP的情况
        self.all_nfps,self.all_divided_nfp,self.all_target_func,self.max_divided_len=[],[],[],0
        for i in range(len(self.polys)):
            if i==index:
                self.all_nfps.append([])
                self.all_target_func.append([])
                self.all_divided_nfp.append([])
                continue

            nfp=self.deleteOnline(self.NFPAssistant.getDirectNFP(self.polys[i],self.polys[index])) # NFP可能有同一直线上的点
            self.all_nfps.append(nfp)
            
            all_bisectior,divided_nfp,target_func=self.getDividedNfp(nfp)
            
            self.all_divided_nfp.append(divided_nfp)
            self.all_target_func.append(target_func)

            if len(divided_nfp)>self.max_divided_len:
                self.max_divided_len=len(divided_nfp)

        # 获得IFR/ifr及其边界情况
        self.ifr=PackingUtil.getInnerFitRectangle(self.polys[index],self.cur_length,self.width)
        self.IFR=Polygon(self.ifr)


if __name__=='__main__':
    # polys=getConvex(num=5)
    polys=getData()
    # nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    # print(datetime.datetime.now(),"计算完成NFP")
    # blf=BottomLeftFill(500,polys,vertical=False,NFPAssistant=nfp_ass)
    # print(blf.polygons)
    LPSearch(500,polys)
