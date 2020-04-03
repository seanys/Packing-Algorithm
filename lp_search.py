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
        self.miu=[[0]*len(self.polys) for _ in range(len(self.polys))] # 用于引导检索

        self.shrink()
        self.updateOverlap() # 更新所有重叠情况

        choose_index=11 # 获得当前最大的Overlap的形状
        self.getPrerequisite(choose_index) # 首先获得全部NFP的拆分
        
        self.getProblemLP() # 获得全部的限制函数

        best_position=self.searchBestPosition() # 获得最优位置
        top_point=self.polys[choose_index][GeoFunc.checkTop(self.polys[choose_index])]
        new_polygon=GeoFunc.getSlide(self.polys[choose_index],best_position[0]-top_point[0],best_position[1]-top_point[1])

        PltFunc.addPolygonColor(new_polygon)
        PltFunc.addPolygonColor(self.ifr)
        self.showPolys()

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
                            # 记录重叠情况
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
                    new_region=new_region.difference(Polygon(self.all_divided_nfp[pair[0]][pair[1]]))
                # 在目标区域增加情况
                if new_region.is_empty!=True and new_region.geom_type!="Point" and new_region.geom_type!="LineString":
                    if new_region.geom_type=="GeometryCollection":
                        self.target_areas[0].append([GeoFunc.collectionToArr(new_region),[i,j]]) # 删除直线/顶点情况
                    else:
                        self.target_areas[0].append([GeoFunc.polyToArr(new_region),[i,j]]) # 最终结果只和顶点相关
                    self.target_function[0].append(self.all_target_func[i][j])

        PltFunc.addPolygon(self.all_divided_nfp[1][3])
        PltFunc.addPolygon(self.all_divided_nfp[4][2])
        PltFunc.addPolygonColor([[255.89339391944443, 462.7381487357385], [340.0, 440.0], [228.0, 248.0], [215.55555555555554, 240.0], [202.401557038374, 240.0], [255.89339391944443, 462.7381487357385]])
        PltFunc.showPlt()
        return
        '''多个形状的计算'''
        for i in range(2,len(self.target_areas)):
        # for i in range(2,3):
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
            
                # 删除重复情况/删除指定的情况
                all_possible_target=PolyListProcessor.deleteRedundancy(all_possible_target)
                all_possible_target=self.deleteTarget(all_possible_target,[item[0] for item in target_area[1:]])

                # 获得判断这些形状是否会重叠，若有则添加并求解目标区域
                new_region=copy.deepcopy(P1)
                for possible_target in all_possible_target:
                    P2=Polygon(self.all_divided_nfp[possible_target[0]][possible_target[1]])
                    inter=P1.intersection(P2)
                    # 添加目标区域/删除上一阶段的结果
                    if inter.area>bias:
                        # 上一阶段区域
                        self.target_areas[i].append([GeoFunc.polyToArr(inter)]+[item for item in target_area[1:]]+[possible_target])
                        new_region.difference(inter)
                        # 目标函数修正
                        [a1,b1,c1],[a2,b2,c2]=self.target_function[i-1][j],self.all_target_func[possible_target[0]][possible_target[1]]
                        self.target_function[i].append([a1+a2,b1+b2,c1+c2])

                # 如果全部删除掉了，就直接清空，否则更新微信的区域
                if new_region.empty==True or new_region.geom_type=="Point" or new_region.geom_type=="LineString":
                    self.target_areas[i-1][j]=[]
                else:
                    target_area[0]=GeoFunc.polyToArr(new_region)

            # 如果该轮没有计算出重叠则停止
            if self.target_areas[i]==[]:
                print("至多",i,"个形状重叠，计算完成")
                break
        
    
    def testModel(self):
        for nfp in self.all_nfps[2:5]:
            PltFunc.addPolygonColor(nfp)

        for i,item in enumerate(self.target_areas[2]):
            if Polygon(item[0]).contains(Point([228.0,248.0]))==True:
                # print(item[1:])
                [a,b,c]=self.target_function[2][i]
                # print(a,b,c)
                # print("综合计算:",a*228+b*248+c)
                value=0
                for target_item in item[1:]:
                    [a,b,c]=self.all_target_func[target_item[0]][target_item[1]]
                    value=value+a*228+b*248+c
                #     print(a*228+b*248+c)
                # print("合并计算:",value)
        # PltFunc.showPlt()

    def searchBestPosition(self):
        '''基于上述获得的区域与目标函数检索最优位置'''
        min_depth,best_position=9999999999,[]
        n=0
        for i,item in enumerate(self.target_areas):
            for j,area_item in enumerate(item):
                a,b,c=self.target_function[i][j]
                for pt in area_item[0]:
                    n=n+1
                    value=a*pt[0]+b*pt[1]+c
                    if pt[0]==228.0 and pt[1]==248.0:
                    #     print("a,b,c:",a,b,c)
                    #     print(value)
                        print(area_item[1:])
                    if value<min_depth and value>bias:
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

if __name__=='__main__':
    starttime = time.time()
    # polys=getConvex(num=5)
    polys=getData()
    # nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    # print(datetime.datetime.now(),"计算完成NFP")
    # blf=BottomLeftFill(500,polys,vertical=False,NFPAssistant=nfp_ass)
    # print(blf.polygons)
    LPSearch(500,polys)
    endtime = time.time()
    print("检索耗时：",(endtime - starttime),"秒")
