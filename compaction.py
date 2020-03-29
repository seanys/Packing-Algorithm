'''
该文件中包含了CollisionFree、Compaction、Separation、SHAH、TOPOS算法
'''
from tools.polygon import GeoFunc,PltFunc,getData,getConvex
from tools.lp import sovleLP,problem
from tools.heuristic import BottomLeftFill
from tools.packing import PolyListProcessor,NFPAssistant
import pandas as pd
import json
from shapely.geometry import Polygon,Point,mapping,LineString
from interval import Interval
import copy
import random
import math

class CollisionFree(object):
    '''
    参考文献：2012 An algorithm for the strip packing problem using collision free region and exact fitting placement
    '''
    def __init__(self,target_poly,polys):
        self.target_poly=target_poly
        self.polys=polys
    


class Compaction(object):
    '''
    参考文献：2006 Solving Irregular Strip Packing problems by hybridising simulated annealing and linear programming
    输入：已经排样好的结果
    输出：n次Compaction后的结果
    '''
    def __init__(self,width,original_polys):
        self.polys=copy.deepcopy(original_polys)
        self.WIDTH=width
        self.DISTANCE=400
        self.getLength()
        self.NFPAssistant=NFPAssistant(self.polys,store_nfp=False,get_all_nfp=False,load_history=True)
        self.run()

    def getLength(self):
        _max=0
        for i in range(0,len(self.polys)):
            extreme_index=GeoFunc.checkRight(self.polys[i])
            extreme=self.polys[i][extreme_index][0]
            if extreme>_max:
                _max=extreme
        self.LENGTH=_max

    # 运行程序
    def run(self):
        # 初始化全部参数，目标参数为z,x1,y1,x2,y...,
        N=len(self.polys)
        a,b,c=[[0]*(2*N+1) for _ in range(9*N+N*N)],[0 for _ in range(9*N+N*N)],[0 for _ in range(N*2+1)]

        # 获得常数限制和多边形的限制
        self.getConstants()
        self.getPolysConstrain()

        # 修改目标函数参数
        c[0]=1

        # 大于所有形状的位置+高度，z-xi>=w OK
        for i in range(N):
            a[i][0],a[i][i*2+1],b[i]=1,-1,self.W[i]
        
        # 限制全部移动距离 OK
        for i in range(N):
            row=N+i*4
            a[row+0][i*2+1],b[row+0]=-1,-self.DISTANCE-self.Xi[i] # -xi>=-DISTANCE-Xi
            a[row+1][i*2+2],b[row+1]=-1,-self.DISTANCE-self.Yi[i] # -yi>=-DISTANCE-Yi
            a[row+2][i*2+1],b[row+2]= 1,-self.DISTANCE+self.Xi[i] # xi>=-DISTANCE+Xi
            a[row+3][i*2+2],b[row+3]= 1,-self.DISTANCE+self.Yi[i] # yi>=-DISTANCE+Yi
        
        # 限制无法移出边界 OK
        for i in range(N):
            row=5*N+i*4
            a[row+0][i*2+1],b[row+0]= 1,self.W_[i] # xi>=Wi*
            a[row+1][i*2+2],b[row+1]= 1,self.H[i]  # yi>=Hi
            a[row+2][i*2+1],b[row+2]=-1,self.W[i]-self.LENGTH  # -xi>=Wi-Length
            a[row+3][i*2+2],b[row+3]=-1,-self.WIDTH  # -yi>=-Width

        # 限制不出现重叠情况 有一点问题
        for i in range(N):
            for j in range(N):
                row=9*N+i*N+j
                if i!=j:
                    a[row][i*2+1],a[row][i*2+2],a[row][j*2+1],a[row][j*2+2],b[row]=self.getOverlapConstrain(i,j)
        
        # 求解计算结果
        result=sovleLP(a,b,c,_type="compaction")

        # 将其转化为坐标，Variable的输出顺序是[x1,..,xn,y1,..,yn,z]
        placement_points=[]
        for i in range(len(result)//2):
            placement_points.append([result[i],result[N+i]])
        
        print("\n初始化高度高度：",self.LENGTH)
        print("优化后高度：",result[N*2])
        
        self.getResult(placement_points)
    
    def getResult(self,placement_points):
        new_polys=[]
        for i,poly in enumerate(self.polys):
            new_polys.append(GeoFunc.getSlide(poly,placement_points[i][0]-self.Xi[i],placement_points[i][1]-self.Yi[i]))
        for i in range(len(self.polys)):
            PltFunc.addPolygon(new_polys[i])
            # PltFunc.addPolygonColor(self.polys[i]) # 初始化的结果
        PltFunc.showPlt(width=self.LENGTH,height=self.LENGTH)
    
    def getOverlapConstrain(self,i,j):
        # 初始化参数
        a_xi,a_yi,a_xj,a_yj,b=0,0,0,0,0
        
        # 获取Stationary Poly的参考点的坐标
        Xi,Yi=self.Xi[i],self.Yi[i] 

        # 获取参考的边
        edge=self.target_edges[i][j] 
        X1,Y1,X2,Y2=edge[0][0],edge[0][1],edge[1][0],edge[1][1]

        # 式1: (y2-y1)*xj+(x1-x2)*yj+x2*y1-x1*y2>0 
        # 式2: (Y2-Y1)*xj+(X1-X2)*yj+X2*Y1-X1*Y2+(xi-Xi)*(Y1-Y2)+(yi-Yi)*(X2-X1)+>0
        # 式3: (Y2-Y1)*xj+(X1-X2)*yj+X2*Y1-X1*Y2+(Y1-Y2)*xi+(X2-X1)*yi-Xi*(Y1-Y2)-Yi*(X2-X1)>0
        # 式4: (Y1-Y2)*xi+(X2-X1)*yi+(Y2-Y1)*xj+(X1-X2)*yj>-X2*Y1+X1*Y2+Xi*(Y1-Y2)+Yi*(X2-X1)
        a_xi,a_yi,x_xj,a_yj,b=Y1-Y2,X2-X1,Y2-Y1,X1-X2,-X2*Y1+X1*Y2+Xi*(Y1-Y2)+Yi*(X2-X1)
        
        return a_xi,a_yi,x_xj,a_yj,b
    
    # 获取所有的常数限制
    def getConstants(self):
        self.W=[] # 最高位置到右侧的距离
        self.W_=[] # 最高位置到左侧的距离
        self.H=[] # 最高点
        self.Xi=[] # Xi的初始位置
        self.Yi=[] # Yi的初始位置
        self.PLACEMENTPOINT=[]
        for i,poly in enumerate(self.polys):
            left_index,bottom_index,right_index,top_index=GeoFunc.checkBound(poly)
            left,bottom,right,top=poly[left_index],poly[bottom_index],poly[right_index],poly[top_index]
            self.PLACEMENTPOINT.append([top[0],top[1]])
            self.Xi.append(top[0])
            self.Yi.append(top[1])
            self.W.append(right[0]-top[0])
            self.W_.append(top[0]-left[0])
            self.H.append(top[1]-bottom[1])
        print("W:",self.W)
        print("W_:",self.W_)
        print("H:",self.H)
        print("Xi:",self.Xi)
        print("Yi:",self.Yi)
        print("PLACEMENTPOINT:",self.PLACEMENTPOINT)
        print("Length:",self.LENGTH)

    # 获取所有两条边之间的关系
    def getPolysConstrain(self):
        self.target_edges=[[0]*len(self.polys) for _ in range(len(self.polys))]
        for i in range(len(self.polys)):
            for j in range(len(self.polys)):
                if i==j:
                    continue
                nfp=self.NFPAssistant.getDirectNFP(self.polys[i],self.polys[j])
                nfp_edges=GeoFunc.getPolyEdges(nfp)
                point=self.PLACEMENTPOINT[j]
                max_distance=-0.00000001
                for edge in nfp_edges:
                    right_distance=self.getRightDistance(edge,point)
                    if right_distance>=max_distance:
                        max_distance=right_distance
                        self.target_edges[i][j]=edge

    @staticmethod
    def getRightDistance(edge,point):
        A=edge[1][1]-edge[0][1]
        B=edge[0][0]-edge[1][0]
        C=edge[1][0]*edge[0][1]-edge[0][0]*edge[1][1]
        D=A*point[0]+B*point[1]+C
        dis=(math.fabs(A*point[0]+B*point[1]+C))/(math.pow(A*A+B*B,0.5))
        if D>0:
            return dis # 右侧返回正
        elif D==0:
            return 0 # 直线上返回0
        else:
            return -dis # 左侧返回负值
    


class Separation(object):
    '''
    参考文献：
    功能：
    '''
    def __init__(self):
        pass

class SHAH(object):
    def __init__(self):
        pass


class TOPOS():
    '''
    参考资料：A beam search implementation for the irregular shape packing problem 
    '''
    def __init__(self):
        pass

if __name__ == "__main__":
    polys=BottomLeftFill(1000,getConvex(num=5)).polygons
    

    # blf = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/blf.csv")
    # polys=json.loads(blf["polys"][1])

    Compaction(1000,polys)
    # ILSQN(poly_list)

