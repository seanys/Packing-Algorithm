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
        self.polys=copy.deepcopy(original_polys)
        self.NFPAssistant=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
        self.getInitialResult()

        self.guidedSearch()        
    
    def guidedSearch(self):
        '''
        实现Guided Search以免陷入局部最优
        '''
        self.miu=[[0]*len(self.polys) for _ in range(len(self.polys))] # 用于引导检索

        self.updateOverlap() # 更新所有重叠情况

        choose_index=0 # 获得当前最大的Overlap的形状
        self.getAllDividedNFP(choose_index) # 首先获得全部NFP的拆分
        self.getConstrain() # 获得全部的限制函数
        
        return 
        # 解决最优问题
        best_position,min_depth=[0,0],999999999
        for problem in self.all_problems:
            res,_min=sovleLP(problem[0],problem[1],problem[2])
            if _min<=min_depth:
                best_position=re

    def updateOverlap(self):
        '''
        1. 获得完整NFP的重叠情况，辅助拆分计算
        2. 获得NFP的重叠面积（未调整）
        '''
        self.overlap,self.overlap_pair=[0 for i in range(len(self.polys))],[[i] for i in range(len(self.polys))]
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                P1,P2=Polygon(self.polys[i]),Polygon(self.polys[j])
                inter=P1.intersection(P2) # 获得重叠区域
                if inter.area>bias:
                    self.overlap[i],self.overlap[j]=self.overlap[i]+inter.area,self.overlap[j]+inter.area
                    self.overlap_pair[i].append(j)
                    self.overlap_pair[j].append(i)

    def searchNewPosition(self,index):
        '''
        为某一个形状寻找更优的位置
        '''
        pass
        # for i in range(N-1):
        #     nfp=self.NFPAssistant.getDirectNFP(self.polys[i],self.polys[N-1])
        #     PltFunc.addPolygon(nfp)
        #     # PltFunc.addPolygon(self.polys[i])
        # # PltFunc.addPolygon(self.polys[N-1])
        # PltFunc.addPolygonColor(PackingUtil.getInnerFitRectangle(self.polys[N-1],450,500))
        # PltFunc.showPlt(width=1400,height=1400)        

    def getAllDividedNFP(self,index):
        '''
        获得全部的NFP的划分情况以及目标函数。由于具体划分较为复杂，简化
        为各角对角线和边界组成的三角形，由于会将会逐一计算，所以我们允许
        拆分后的结果重叠
        '''
        self.allDividedNFP=[]

        # 遍历全部的形状
        for i in range(1):
            nfp=self.deleteOnline(self.NFPAssistant.getDirectNFP(self.polys[4],self.polys[5])) # NFP可能有同一直线上的点
            
            # 遍历NFP的所有顶点计算角平分线
            all_bisectior=[]
            for i in range(-2,len(nfp)-2):
                vec=self.getAngularBisector(nfp[i],nfp[i+1],nfp[i+2])
                all_bisectior.append([nfp[i+1],[nfp[i+1][0]+vec[0]*1000,nfp[i+1][1]+vec[1]*1000]])

            # 计算全部的三角形区域和附带边
            dividedNFP=[]
            for i in range(-1,len(all_bisectior)-1):
                line1,line2=all_bisectior[i],all_bisectior[i+1]
                inter=self.lineIntersection(line1,line2)
                dividedNFP.append([nfp[i],nfp[i+1],inter]) # [边界点1,边界点2,交点]
            
            self.allDividedNFP.append(dividedNFP)
            

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



    def getConstrain(self):
        '''
        基于NFP获得全部的约束
        '''
        self.all_problems=[]
        self.getDivededOverlap() # 获得全部的重叠情况

        # 获得拆分后的NFP的重叠
        
        # 获得三个区域重叠

        # 获得四个区域重叠

        # 重叠转




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

