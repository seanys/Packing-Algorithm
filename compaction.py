"""
该文件实现了拆分算法 Separation去除重叠
和Compaction压缩当前解
-----------------------------------
Created on Wed Dec 11, 2020
@author: seanys,prinway
-----------------------------------
"""
from tools.polygon import GeoFunc,PltFunc
from tools.lp import sovleLP,problem
import pandas as pd
import json
from shapely.geometry import Polygon,Point,mapping,LineString
from interval import Interval
import copy
import random
import math
from tools.lp_assistant import LPAssistant
    
class CollisionFree(object):
    '''
    参考文献：2012 An algorithm for the strip packing problem using collision free region and exact fitting placement
    '''
    def __init__(self,target_poly,polys):
        self.target_poly=target_poly
        self.polys=polys
    
class SHAH(object):
    def __init__(self):
        pass



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
        result,_=sovleLP(a,b,c,_type="compaction")

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

        '''
        式1: (y2-y1)*xj+(x1-x2)*yj+x2*y1-x1*y2>0 
        式2: (Y2-Y1)*xj+(X1-X2)*yj+X2*Y1-X1*Y2+(xi-Xi)*(Y1-Y2)+(yi-Yi)*(X2-X1)+>0
        式3: (Y2-Y1)*xj+(X1-X2)*yj+X2*Y1-X1*Y2+(Y1-Y2)*xi+(X2-X1)*yi-Xi*(Y1-Y2)-Yi*(X2-X1)>0
        式4: (Y1-Y2)*xi+(X2-X1)*yi+(Y2-Y1)*xj+(X1-X2)*yj>-X2*Y1+X1*Y2+Xi*(Y1-Y2)+Yi*(X2-X1)
        '''
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
    参考文献：Solving Irregular Strip Packing problems by hybridising simulated annealing and linear programming
    功能：拆分全部的重叠
    '''
    def __init__(self,polys,poly_status,width,length):
        self.all_nfp=pd.read_csv("/Users/sean/Documents/Projects/Data/fu_simplify.csv")
        self.poly_status=copy.deepcopy(poly_status)
        self.polys=copy.deepcopy(polys)
        self.WIDTH=width
        # print("初始高度:",LPAssistant.getLength(polys))
        # self.LENGTH=LPAssistant.getLength(polys)+32
        self.LENGTH=length
        self.DISTANCE=400
        # PltFunc.showPolys(self.polys)
        self.main()
        
    def main(self):
        # 初始化全部参数，目标参数为z,x1,y1,x2,y...,
        N=len(self.polys)
        a,b,c=[[0]*(2*N+N*N) for _ in range(8*N+N*N)],[0 for _ in range(8*N+N*N)],[0 for _ in range(N*2+N*N)]
        
        # 获得常数限制和多边形的限制
        self.getConstants()
        self.getTargetEdges()
        
        # 限制全部移动距离 OK
        for i in range(N):
            row=i*4
            a[row+0][i*2+0],b[row+0]=-1,-self.DISTANCE-self.Xi[i] # -xi>=-DISTANCE-Xi
            a[row+1][i*2+1],b[row+1]=-1,-self.DISTANCE-self.Yi[i] # -yi>=-DISTANCE-Yi
            a[row+2][i*2+0],b[row+2]= 1,-self.DISTANCE+self.Xi[i] # xi>=-DISTANCE+Xi
            a[row+3][i*2+1],b[row+3]= 1,-self.DISTANCE+self.Yi[i] # yi>=-DISTANCE+Yi
        
        # 限制无法移出边界 OK
        for i in range(N):
            row=4*N+i*4
            a[row+0][i*2+0],b[row+0]= 1,self.W_[i] # xi>=Wi*
            a[row+1][i*2+1],b[row+1]= 1,self.H[i]  # yi>=Hi
            a[row+2][i*2+0],b[row+2]=-1,self.W[i]-self.LENGTH  # -xi>=Wi-Length
            a[row+3][i*2+1],b[row+3]=-1,-self.WIDTH  # -yi>=-Width

        # 限制不出现重叠情况 有一点问题
        for i in range(N):
            for j in range(N):
                row=8*N+i*N+j
                if i!=j:
                    a[row][i*2+0],a[row][i*2+1],a[row][j*2+0],a[row][j*2+1],b[row]=self.getOverlapConstrain(i,j)
                    a[row][2*N+i*N+j],c[2*N+i*N+j]=1,1 # 目标函数变化 
                else:
                    a[row][2*N+i*N+j],c[2*N+i*N+j],b[row]=1,1,0 
        
        # 求解计算结果
        result,self.final_value=sovleLP(a,b,c,_type="separation")

        # 将其转化为坐标，Variable的输出顺序是[a00,..,ann,x1,..,xn,y1,..,yn]
        placement_points=[]
        # print(len(result))
        for i in range(N*N,N*N+N):
            placement_points.append([result[i],result[i+N]])
                
        self.getResult(placement_points)
    
    def getResult(self,placement_points):
        self.final_polys,self.final_poly_status=[],copy.deepcopy(self.poly_status)
        for i,poly in enumerate(self.polys):
            self.final_polys.append(GeoFunc.getSlide(poly,placement_points[i][0]-self.Xi[i],placement_points[i][1]-self.Yi[i]))
            self.final_poly_status[i][1]=[placement_points[i][0],placement_points[i][1]]
        # for i in range(len(self.polys)):
        #     PltFunc.addPolygon(new_polys[i])
            # PltFunc.addPolygonColor(self.polys[i]) # 初始化的结果
        # PltFunc.showPlt(width=1500,height=1500)
            
    def getOverlapConstrain(self,i,j):
        # 初始化参数
        a_xi,a_yi,a_xj,a_yj,b=0,0,0,0,0
        
        # 获取Stationary Poly的参考点的坐标
        Xi,Yi=self.Xi[i],self.Yi[i] 

        # 获取参考的边
        edge=self.target_edges[i][j] 
        X1,Y1,X2,Y2=edge[0][0],edge[0][1],edge[1][0],edge[1][1]

        '''
        非重叠情况
        式1: (y2-y1)*xj+(x1-x2)*yj+x2*y1-x1*y2>0  右侧距离大于0
        式2: (Y2-Y1)*xj+(X1-X2)*yj+X2*Y1-X1*Y2+(xi-Xi)*(Y1-Y2)+(yi-Yi)*(X2-X1)+>0
        式3: (Y2-Y1)*xj+(X1-X2)*yj+X2*Y1-X1*Y2+(Y1-Y2)*xi+(X2-X1)*yi-Xi*(Y1-Y2)-Yi*(X2-X1)>0
        式4: (Y1-Y2)*xi+(X2-X1)*yi+(Y2-Y1)*xj+(X1-X2)*yj>-X2*Y1+X1*Y2+Xi*(Y1-Y2)+Yi*(X2-X1)
        重叠情况
        式1: -((y2-y1)*xj+(x1-x2)*yj+x2*y1-x1*y2)-a_ij<0  左侧距离小于0
        式2: (y2-y1)*xj+(x1-x2)*yj+x2*y1-x1*y2+a_ij>0
        式1: (Y1-Y2)*xi+(X2-X1)*yi+(Y2-Y1)*xj+(X1-X2)*yj+a_ij>-X2*Y1+X1*Y2+Xi*(Y1-Y2)+Yi*(X2-X1) 左侧距离小于0
        总结: 重叠的时候由于求出来是负值，最终只增加了一个a_ij，参数肯定是1
        '''
        a_xi,a_yi,a_xj,a_yj,b=Y1-Y2,X2-X1,Y2-Y1,X1-X2,-X2*Y1+X1*Y2+Xi*(Y1-Y2)+Yi*(X2-X1)
        
        return a_xi,a_yi,a_xj,a_yj,b
    
    # 获取所有的常数限制
    def getConstants(self):
        self.W=[] # 最高位置到右侧的距离
        self.W_=[] # 最高位置到左侧的距离
        self.H=[] # 最高点
        self.Xi=[] # Xi的初始位置
        self.Yi=[] # Yi的初始位置
        self.PLACEMENTPOINT=[]
        for i,poly in enumerate(self.polys):
            left,bottom,right,top=LPAssistant.getBoundPoint(poly)
            self.PLACEMENTPOINT.append([top[0],top[1]])
            self.Xi.append(top[0])
            self.Yi.append(top[1])
            self.W.append(right[0]-top[0])
            self.W_.append(top[0]-left[0])
            self.H.append(top[1]-bottom[1])
        # print("W:",self.W)
        # print("W_:",self.W_)
        # print("H:",self.H)
        # print("Xi:",self.Xi)
        # print("Yi:",self.Yi)
        # print("PLACEMENTPOINT:",self.PLACEMENTPOINT)
        # print("Length:",self.LENGTH)

    # 获取所有两条边之间的关系
    def getTargetEdges(self):
        self.target_edges=[[0]*len(self.polys) for _ in range(len(self.polys))]
        self.overlap_pair=[[False]*len(self.polys) for _ in range(len(self.polys))]
        for i in range(len(self.polys)):
            for j in range(len(self.polys)):
                if i==j:
                    continue
                nfp=self.getNFP(i,j)
                nfp_edges=GeoFunc.getPolyEdges(nfp)
                point=self.PLACEMENTPOINT[j]
                if Polygon(nfp).contains(Point(point)):
                    # 如果包含，则寻找距离最近的那个
                    self.overlap_pair[i][j]=True
                    min_distance=99999999999999
                    for edge in nfp_edges:
                        left_distance=-self.getRightDistance(edge,point)
                        if left_distance<=min_distance:
                            min_distance=left_distance
                            self.target_edges[i][j]=copy.deepcopy(edge)
                else:
                    max_distance=-0.00000001
                    for edge in nfp_edges:
                        right_distance=self.getRightDistance(edge,point)
                        if right_distance>=max_distance:
                            max_distance=right_distance
                            self.target_edges[i][j]=copy.deepcopy(edge)

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

    def getNFP(self,j,i):
        # j是固定位置，i是移动位置
        row=j*192+i*16+self.poly_status[j][2]*4+self.poly_status[i][2]
        bottom_pt=LPAssistant.getBottomPoint(self.polys[j])
        delta_x,delta_y=bottom_pt[0],bottom_pt[1]
        nfp=GeoFunc.getSlide(json.loads(self.all_nfp["nfp"][row]),delta_x,delta_y)
        return nfp

def searchForBest(polys,poly_status,width,length):
    # 记录最优结果
    best_poly_status,best_polys=[],[]
    cur_length=length

    # 循环检索最优位置(Polys不需要变化)
    while True:
        print("允许高度:",cur_length)
        result_polys,result_poly_status,result_value=searchOneLength(polys,poly_status,width,cur_length)
        if result_value==0:
            best_polys=result_polys
            best_poly_status=result_poly_status
            break
        cur_length=cur_length+4
    
    print("开始准确检索")
    # 精准检索最优结果
    for i in range(3):
        cur_length=cur_length-1
        print("允许高度:",cur_length)
        result_polys,result_poly_status,result_value=searchOneLength(polys,poly_status,width,cur_length)
        if result_value!=0:
            break
        best_poly_status=result_poly_status
        best_polys=result_polys

    best_length=cur_length+1
    print("最终高度:",best_length)
    PltFunc.showPolys(best_polys)
    # 执行Compaction代码

def searchOneLength(polys,poly_status,width,length):
    input_polys=copy.deepcopy(polys) # 每次输入的形状
    last_value=99999999999
    final_polys,final_poly_status=[],[]
    while True:
        res=Separation(input_polys,poly_status,width,length)
        # 如果没有重叠，或者等于上一个状态
        if res.final_value==0 or abs(res.final_value-last_value)<0.001:
            last_value=res.final_value
            final_polys=copy.deepcopy(res.final_polys)
            final_poly_status=copy.deepcopy(res.final_poly_status)
            break
        # 如果有变化，则更换状态再试一次
        input_polys=copy.deepcopy(res.final_polys)
        last_value=res.final_value
    return final_polys,final_poly_status,last_value


if __name__ == "__main__":
    blf = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/blf.csv")
    index=7
    polys,poly_status,width=json.loads(blf["polys"][index]),json.loads(blf["poly_status"][index]),int(blf["width"][index])
    # Separation(polys,poly_status,width)
    searchForBest(polys,poly_status,width,628.1533587455999)

    Compaction(polys,poly_status,width)

    # ILSQN(poly_list)

