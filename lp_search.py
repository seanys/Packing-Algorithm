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
        
        return 
        
        self.getConstrain() # 获得全部的限制函数
        
        return 

        # 解决最优问题
        best_position,min_depth=[0,0],999999999
        for problem in self.all_problems:
            res,_min=sovleLP(problem[0],problem[1],problem[2])
            if _min<=min_depth:
                best_position=re

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
        self.all_nfps,self.all_divided_nfp,self.max_divided_len=[],[],0
        for i in range(len(self.polys)):
            if i==index:
                self.all_nfps.append([])
                continue

            nfp=self.deleteOnline(self.NFPAssistant.getDirectNFP(self.polys[i],self.polys[index])) # NFP可能有同一直线上的点
            self.all_nfps.append(nfp)
            
            # 遍历NFP的所有顶点计算角平分线
            all_bisectior=[]
            for i in range(-2,len(nfp)-2):
                vec=self.getAngularBisector(nfp[i],nfp[i+1],nfp[i+2])
                all_bisectior.append([nfp[i+1],[nfp[i+1][0]+vec[0]*1000,nfp[i+1][1]+vec[1]*1000]])

            # 计算全部的三角形区域和附带边
            divided_nfp=[]
            for i in range(-1,len(all_bisectior)-1):
                line1,line2=all_bisectior[i],all_bisectior[i+1]
                inter=self.lineIntersection(line1,line2)
                divided_nfp.append([nfp[i-1],nfp[i],inter]) # [边界点1,边界点2,交点]
            
            self.all_divided_nfp.append(divided_nfp)

            if len(divided_nfp)>self.max_divided_len:
                self.max_divided_len=len(divided_nfp)

        # 获得IFR/ifr及其边界情况
        self.ifr=PackingUtil.getInnerFitRectangle(self.polys[index],self.cur_length,self.width)
        self.IFR=Polygon(self.ifr)

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
    def getConstrain(self):
        self.updateNFPOverlap() # 更新NFP的重叠情况
        self.target_areas=[[],[],[],[],[],[],[]] # 用于存储目标区域，最高允许6个形状重叠

        '''首先获得全部的一对对的NFP的结果，以及目标函数'''
        self.pair_nfps=[[[] for j in range(self.max_divided_len)] for i in range(len(self.polys))]
        for i in range(len(self.nfp_overlap_pair)):
            for j in self.nfp_overlap_pair[i][1:]:
                # 逐一计算对应的小形状
                for m,divided_nfp_i in enumerate(self.all_divided_nfp[i]):
                    for n,divided_nfp_j in enumerate(self.all_divided_nfp[j]):
                        overlap,overlap_poly=self.polysOverlapIFR(divided_nfp_i,divided_nfp_j)
                        # 重叠则记录对应的细分重叠
                        if overlap==True:
                            self.pair_nfps[i][m].append([j,n])
                        # 重叠且j>i则存储重叠情况，避免重复
                        if overlap==True and j>i:
                            self.target_areas[1].append([overlap_poly,[i,m],[j,n]])

        '''然后获得删除重叠区域以及IFR范围的区间'''
        for i,nfp_divided in enumerate(self.all_divided_nfp):
            for j,item in enumerate(nfp_divided):
                # 删除与IFR重叠区域
                new_region=Polygon(item).intersection(self.IFR)
                # 删除与指定重叠区域
                for pair in self.pair_nfps[i][j]:
                    new_region=new_region.difference(Polygon(self.all_divided_nfp[pair[0]][pair[1]]))
                if new_region.is_empty!=True and new_region.geom_type!="Point" and new_region.geom_type!="LineString":
                    print(new_region)
                    self.target_areas[0].append([GeoFunc.polyToArr(new_region),i,j]) # 最终结果只和顶点相关
        
        '''多个形状的计算'''
        for i in range(1,2):
        # for i in range(1,len(self.target_areas)-1):
            for j,target_area in enumerate(self.target_areas[i]):
                # 获得当前目标可行解
                area,P1=target_area[0],Polygon(target_area[0])

                # 其他可能有的区域
                all_possible_target=[]
                for item in target_area[1:]:                    
                    all_possible_target=all_possible_target+self.pair_nfps[item[0]][item[1]]
            
                # 删除重复情况/删除指定的情况
                all_possible_target=PolyListProcessor.deleteRedundancy(all_possible_target)
                all_possible_target=self.delteTarget(all_possible_target,[item[0] for item in target_area[1:]])

                # 获得判断这些形状是否会重叠，若有则添加并求解目标区域
                new_region=copy.deepcopy(P1)
                for possible_target in all_possible_target:
                    P2=Polygon(self.all_divided_nfp[possible_target[0]][possible_target[1]])
                    inter=P1.intersection(P2)
                    # 添加目标区域/删除上一阶段的结果
                    if inter.area>bias:
                        self.target_areas[i+1].append([GeoFunc.polyToArr(inter),target_area[1],target_area[2],possible_target])
                        new_region.difference(inter)

                # 如果全部删除掉了，就直接清空，否则更新微信的区域
                if new_region.empty==True or new_region.geom_type=="Point" or new_region.geom_type=="LineString":
                    self.target_areas[i][j]=[]
                else:
                    target_area[0]=GeoFunc.polyToArr(new_region)
            # 如果该轮没有计算出重叠则停止
            if self.target_areas[i+1]==[]:
                break
    
    
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
    
    def delteTarget(self,_list,target):
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

    # 多边形转化为限制
    def convertPolyToConstrain(self,poly):
        '''将多边形转化为区域（凸多边形）'''
        edges=GeoFunc.getPolyEdges(polys)
        constrain=[]
        for i in range(-1,len(poly)-1):
            # 初始化计算
            x1,y2,x2,y2=poly[i][0],poly[i][1],poly[i+1][0],poly[i+1][1]
            A,B,C=y2-y1,x1-x2,x2*y1-x1*y2
            # 向量为1/4象限大于0，2/3为小于0，需要取负
            if x2-x1>0:
                constrain.append([A,B,C])
            else:
                constrain.append([-A,-B,-C])
        return constrain

if __name__=='__main__':
    # polys=getConvex(num=5)
    polys=getData()
    # nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    # print(datetime.datetime.now(),"计算完成NFP")
    # blf=BottomLeftFill(500,polys,vertical=False,NFPAssistant=nfp_ass)
    # print(blf.polygons)
    LPSearch(1000,polys)

