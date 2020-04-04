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

bias=0.0000001

class LPSearch(object):
    '''
    线性检索算法，采用数据集Fu
    '''
    def __init__(self,width,original_polys):
        self.width=width
        self.polys=copy.deepcopy(original_polys)
        self.NFPAssistant=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
        self.getInitialResult()

        self.guidedSearch()        
    
    def guidedSearch(self):
        '''
        实现Guided Search以免陷入局部最优
        '''
        starttime = time.time()
        
        self.miu=[[0]*len(self.polys) for _ in range(len(self.polys))] # 用于引导检索

        self.shrink()
        self.updateOverlap() # 更新所有重叠情况

        choose_index=11 # 获得当前最大的Overlap的形状
        self.getPrerequisite(choose_index) # 首先获得全部NFP的拆分
        
        self.getProblemLP() # 获得全部的限制函数

        best_position=self.searchBestPosition() # 获得最优位置

        endtime = time.time()
        print("检索耗时：",(endtime - starttime),"秒")

        # top_point=self.polys[choose_index][GeoFunc.checkTop(self.polys[choose_index])]
        # new_polygon=GeoFunc.getSlide(self.polys[choose_index],best_position[0]-top_point[0],best_position[1]-top_point[1])

        # PltFunc.addPolygonColor(new_polygon)
        # PltFunc.addPolygonColor(self.ifr)
        # self.showPolys()

    def updateOverlap(self):
        '''获得的重叠面积（未调整）'''
        self.overlap=[0 for i in range(len(self.polys))]
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                P1,P2=Polygon(self.polys[i]),Polygon(self.polys[j])
                inter=P1.intersection(P2) # 获得重叠区域
                if inter.area>bias:
                    self.overlap[i],self.overlap[j]=self.overlap[i]+inter.area,self.overlap[j]+inter.area
    
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
    
    # 获得全部的NFP及其拆分结果，获得IFR和限制
    def getPrerequisite(self,index):
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
            
            # 遍历NFP的所有顶点计算角平分线
            all_bisectior=[]
            for i in range(-2,len(nfp)-2):
                vec=self.getAngularBisector(nfp[i],nfp[i+1],nfp[i+2])
                all_bisectior.append([nfp[i+1],[nfp[i+1][0]+vec[0]*1000,nfp[i+1][1]+vec[1]*1000]])

            # 计算全部的三角形区域和附带边
            divided_nfp,target_func=[],[]
            for i in range(-1,len(all_bisectior)-1):
                line1,line2=all_bisectior[i],all_bisectior[i+1]
                inter=self.lineIntersection(line1,line2)
                divided_nfp.append([nfp[i-1],nfp[i],inter]) # [边界点1,边界点2,交点]
                target_func.append(self.getTargetFunction([nfp[i-1],nfp[i]]))
            
            self.all_divided_nfp.append(divided_nfp)
            self.all_target_func.append(target_func)

            if len(divided_nfp)>self.max_divided_len:
                self.max_divided_len=len(divided_nfp)

        # 获得IFR/ifr及其边界情况
        self.ifr=PackingUtil.getInnerFitRectangle(self.polys[index],self.cur_length,self.width)
        self.IFR=Polygon(self.ifr)
    
    def showPolys(self):
        for poly in self.polys:
            PltFunc.addPolygon(poly)
        PltFunc.addPolygonColor([[0,0],[self.cur_length,0],[self.cur_length,self.width],[0,self.width]])
        PltFunc.showPlt(width=1500,height=1500)
    
    # 判断内外侧
    def judgeInner(self,edge):
        '''
        （凹多边形）判断内侧是在左边还是右边  
        '''
        # for 
        pass

    # 基于NFP获得全部的约束
    def getProblemLP(self):
        self.updateNFPOverlap() # 更新NFP的重叠情况
        self.target_areas=[[],[],[],[],[],[],[],[]] # 用于存储目标区域，最高允许8个形状重叠
        self.target_function=[[],[],[],[],[],[],[],[]] # 用于计算目标函数，分别为xy的参数

        '''首先获得全部的一对对的NFP的结果，以及目标函数'''
        self.pair_nfps=[[[] for j in range(self.max_divided_len)] for i in range(len(self.polys))]
        for i in range(len(self.nfp_overlap_pair)):
            for j in self.nfp_overlap_pair[i][1:]:
                if j<i:
                    continue
                # 逐一计算对应的小形状
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
        
        '''然后获得删除重叠区域以及IFR范围的区间'''
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


        '''多个形状的计算'''
        for i in range(2,len(self.target_areas)):
            # 遍历上一阶段计算的结果
            for j,target_area in enumerate(self.target_areas[i-1]):
                # 获得当前目标可行解
                area,P1=target_area[0],Polygon(target_area[0])

                # 其他可能有的区域
                all_possible_target=[]
                if i>=3:
                    # 如果是四个形状重叠，只需要增加计算三个形状中最后加入的
                    all_possible_target=self.pair_nfps[target_area[-1][0]][target_area[-1][1]]
                else:
                    # 需要计算其他两个
                    for item in target_area[1:]:
                        all_possible_target=all_possible_target+self.pair_nfps[item[0]][item[1]]
            
                # 删除重复情况
                all_possible_target=PolyListProcessor.deleteRedundancy(all_possible_target)

                # 删除所有更小的，保证正序，获得判断这些形状是否会重叠，若有则添加并求解目标区域
                all_possible_target_larger=self.deleteTarget(all_possible_target,[i for i in range(0,max(item[0] for item in target_area[1:])+1)])
                for possible_target in all_possible_target_larger:
                    P2=Polygon(self.all_divided_nfp[possible_target[0]][possible_target[1]])
                    # 只有相交才进一步计算
                    if P1.intersects(P2):
                        inter=P1.intersection(P2)
                        if inter.area>bias:
                            self.target_areas[i].append([self.processRegion(inter)]+[item for item in target_area[1:]]+[possible_target])
                            [a1,b1,c1],[a2,b2,c2]=self.target_function[i-1][j],self.all_target_func[possible_target[0]][possible_target[1]]
                            self.target_function[i].append([a1+a2,b1+b2,c1+c2])
                
                # 删除已经有的，遍历计算重叠
                new_region=copy.deepcopy(P1)
                all_possible_target_difference=self.deleteTarget(all_possible_target,[item[0] for item in target_area[1:]])
                for difference_target in all_possible_target_difference:
                    P2=Polygon(self.all_divided_nfp[difference_target[0]][difference_target[1]])
                    if new_region.intersects(P2):
                        new_region=new_region.difference(P2)

                if new_region.is_empty!=True and new_region.geom_type!="Point" and new_region.geom_type!="LineString" and new_region.area>bias:
                    target_area[0]=self.processRegion(new_region)
                else:
                    self.target_areas[i-1][j]=[]

            # 如果该轮没有计算出重叠则停止
            if self.target_areas[i]==[]:
                print("至多",i,"个形状重叠，计算完成")
                break
              
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
        return best_position
    
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

    def delteEmpty(self,_list):
        new_list=[]
        for item in _list:
            if item!=[]:
                new_list.append(item)
        return new_list

    @staticmethod
    def lineIntersection(line1, line2):
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
            if vec1[0]!=vec2[0]:
                new_poly.append(poly[i+1])
        return new_poly
    
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

    # 收缩边界
    def shrink(self):
        self.cur_length=self.length*0.95
        # for 

    # 获得宽度    
    def getLength(self):
        _max=0
        for i in range(0,len(self.polys)):
            extreme_index=GeoFunc.checkRight(self.polys[i])
            extreme=self.polys[i][extreme_index][0]
            if extreme>_max:
                _max=extreme
        self.length=_max
        print("高度:",_max)

    # 获得初始解
    def getInitialResult(self):
        blf = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/blf.csv")
        self.polys=json.loads(blf["polys"][2])
        print("一共",len(self.polys),"个形状")
        self.getLength()
    
    def processRegion(self,region):
        area=[]
        if region.geom_type=="Polygon":
            area=GeoFunc.polyToArr(region)  # 最终结果只和顶点相关
        else:
            for shapely_item in list(region):
                if shapely_item.area>bias:
                    area=area+GeoFunc.polyToArr(shapely_item)
        return area

    # 获得角平分线
    def getAngularBisector(self,pt1,pt2,pt3):
        '''
        输入：pt1/pt3为左右两个点，pt2为中间的点
        输出：该角的对角线
        '''
        vec1=self.getDirectionalVector([pt1[0]-pt2[0],pt1[1]-pt2[1]])
        vec2=self.getDirectionalVector([pt3[0]-pt2[0],pt3[1]-pt2[1]])
        bisector=[]
        if vec1[0]+vec2[0]==0:
            new_x=math.cos(math.pi/2)*vec1[0] - math.sin(math.pi/2)*vec1[1]
            new_y=math.cos(math.pi/2)*vec1[1] + math.sin(math.pi/2)*vec1[0]
            bisector=[new_x,new_y] # 获得垂直方向，默认选择[pt2,pt3]的右侧
        else:
            bisector=[(vec1[0]+vec2[0]),vec1[1]+vec2[1]] # 获得对角方向，长度为sqrt(2)
        return bisector

if __name__=='__main__':
    # polys=getConvex(num=5)
    polys=getData()
    # nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    # print(datetime.datetime.now(),"计算完成NFP")
    # blf=BottomLeftFill(500,polys,vertical=False,NFPAssistant=nfp_ass)
    # print(blf.polygons)
    LPSearch(500,polys)
