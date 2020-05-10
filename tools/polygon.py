from shapely.geometry import Polygon,Point,mapping,LineString
from shapely.ops import unary_union
from shapely import affinity
import pyclipper 
import math
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import csv
import time
import logging
import random
import copy
import os
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(filename)s-line%(lineno)d:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")

bias=0.000001

class Poly(object):
    '''
    用于后续的Poly对象
    '''
    def __init__(self,num,poly,allowed_rotation):
        self.num=num
        self.poly=poly
        self.cur_poly=poly
        self.allowed_rotation=[0,180]

class GeoFunc(object):
    '''
    几何相关函数
    1. checkBottom、checkTop、checkLeft、checkRight暂时不考虑多个点
    2. checkBottom和checkLeft均考虑最左下角
    '''
    def almostContain(line,point):
        # 会由于int导致计算偏差！！！！！！
        pt1=[line[0][0],line[0][1]]
        pt2=[line[1][0],line[1][1]]
        point=[point[0],point[1]]

        # 水平直线情况：通过比较两个点和中间点比较
        if abs(pt1[1]-point[1])<bias and abs(pt2[1]-point[1])<bias:
            # print("水平情况")
            if (pt1[0]-point[0])*(pt2[0]-point[0])<0:
                return True
            else:
                return False
    
        # 排除垂直的情况
        if abs(pt1[0]-point[0])<bias and abs(pt2[0]-point[0])<bias:
            # print("垂直情况")
            if (pt1[1]-point[1])*(pt2[1]-point[1])<0:
                return True
            else:
                return False

        if abs(pt1[0]-point[0])<bias or abs(pt2[0]-point[0])<bias or abs(pt1[0]-pt2[0])<bias:
            return False
        
        # 正常情况，计算弧度的差值
        arc1=np.arctan((line[0][1]-line[1][1])/(line[0][0]-line[1][0]))
        arc2=np.arctan((point[1]-line[1][1])/(point[0]-line[1][0]))
        if abs(arc1-arc2)<bias: # 原值0.03，dighe近似平行修正为0.01
            if (point[1]-pt1[1])*(pt2[1]-point[1])>0 and (point[0]-pt1[0])*(pt2[0]-point[0])>0:
                # print("一般情况")
                return True
            else:
                return False
        else:
            return False
    
    def computeInterArea(orginal_inter):
        '''
        计算相交区域的面积
        '''
        inter=mapping(orginal_inter)
        # 一个多边形
        if inter["type"]=="Polygon":
            if len(inter["coordinates"])>0:
                poly=inter["coordinates"][0]
                return Polygon(poly).area
            else: return 0
        if inter["type"]=="MultiPolygon":
            area=0
            for _arr in inter["coordinates"]:
                poly=_arr[0]
                area=area+Polygon(poly).area
            return area
        
        if inter["type"]=="GeometryCollection":
            area=0
            for _arr in inter["geometries"]:
                if _arr["type"]=="Polygon":
                    poly=_arr["coordinates"][0]
                    area=area+Polygon(poly).area
            return area
        return 0

    def checkBottom(poly):
        polyP=Polygon(poly)
        min_y=polyP.bounds[1]
        for index,point in enumerate(poly):
            if point[1]==min_y:
                return index

    def checkTop(poly):
        polyP=Polygon(poly)
        max_y=polyP.bounds[3]
        for index,point in enumerate(poly):
            if point[1]==max_y:
                return index
    
    def checkLeft(poly):
        polyP=Polygon(poly)
        min_x=polyP.bounds[0]
        for index,point in enumerate(poly):
            if point[0]==min_x:
                return index
    
    def checkRight(poly):
        polyP=Polygon(poly)
        max_x=polyP.bounds[2]
        for index,point in enumerate(poly):
            if point[0]==max_x:
                return index

    def checkBound(poly):
        return GeoFunc.checkLeft(poly), GeoFunc.checkBottom(poly), GeoFunc.checkRight(poly), GeoFunc.checkTop(poly)
    
    def checkBoundPt(poly):
        '''获得边界的点'''
        left,bottom,right,top=poly[0],poly[0],poly[0],poly[0]
        for i,pt in enumerate(poly):
            if pt[0]<left[0]:
                left=pt
            if pt[0]>right[0]:
                right=pt
            if pt[1]>top[1]:
                top=pt
            if pt[1]<bottom[1]:
                bottom=pt
        return left,bottom,right,top

    def checkBoundValue(poly):
        '''获得边界的值'''
        left,bottom,right,top=poly[0][0],poly[0][1],poly[0][0],poly[0][1]
        for i,pt in enumerate(poly):
            if pt[0]<left:
                left=pt[0]
            if pt[0]>right:
                right=pt[0]
            if pt[1]>top:
                top=pt[1]
            if pt[1]<bottom:
                bottom=pt[1]
        return left,bottom,right,top

    def slideToPoint(poly,pt1,pt2):
        GeoFunc.slidePoly(poly,pt2[0]-pt1[0],pt2[1]-pt1[1])

    def getSlide(poly,x,y):
        '''
        获得平移后的情况
        '''
        new_vertex=[]
        for point in poly:
            new_point=[point[0]+x,point[1]+y]
            new_vertex.append(new_point)
        return new_vertex

    def slidePoly(poly,x,y):
        for point in poly:
            point[0]=point[0]+x
            point[1]=point[1]+y

    def polyToArr(inter):
        res=mapping(inter)
        _arr=[]
        if res["type"]=="MultiPolygon":
            for poly in res["coordinates"]:
                for point in poly[0]:
                    _arr.append([point[0],point[1]])
        elif res["type"]=="GeometryCollection":
            for item in res["geometries"]:
                if item["type"]=="Polygon":
                    for point in item["coordinates"][0]:
                        _arr.append([point[0],point[1]])
        else:
            if res["coordinates"][0][0]==res["coordinates"][0][-1]:
                for point in res["coordinates"][0][0:-1]:
                    _arr.append([point[0],point[1]])
            else:
                for point in res["coordinates"][0]:
                    _arr.append([point[0],point[1]])
        return _arr

    def normData(poly,num):
        for ver in poly:
            ver[0]=ver[0]*num
            ver[1]=ver[1]*num

    '''近似计算'''
    def crossProduct(vec1,vec2):
        res=vec1[0]*vec2[1]-vec1[1]*vec2[0]
        # 最简单的计算
        if abs(res)<bias:
            return 0
        # 部分情况叉积很大但是仍然基本平行
        if abs(vec1[0])>bias and abs(vec2[0])>bias:
            if abs(vec1[1]/vec1[0]-vec2[1]/vec2[0])<bias:
                return 0
        return res
    
    '''用于touching计算交点 可以与另一个交点计算函数合并'''
    def intersection(line1,line2):
        # 如果可以直接计算出交点
        Line1=LineString(line1)
        Line2=LineString(line2)
        inter=Line1.intersection(Line2)
        if inter.is_empty==False:
            mapping_inter=mapping(inter)
            if mapping_inter["type"]=="LineString":
                inter_coor=mapping_inter["coordinates"][0]
            else:
                inter_coor=mapping_inter["coordinates"]
            return inter_coor

        # 对照所有顶点是否相同
        res=[]
        for pt1 in line1:
            for pt2 in line2:
                if GeoFunc.almostEqual(pt1,pt2)==True:
                    # print("pt1,pt2:",pt1,pt2)
                    res=pt1
        if res!=[]:
            return res

        # 计算是否存在almostContain
        for pt in line1:
            if GeoFunc.almostContain(line2,pt)==True:
                return pt
        for pt in line2:
            if GeoFunc.almostContain(line1,pt)==True:
                return pt
        return []
    
    ''' 主要用于判断是否有直线重合 过于复杂需要重构'''
    def newLineInter(line1,line2):
        vec1=GeoFunc.lineToVec(line1)
        vec2=GeoFunc.lineToVec(line2)
        vec12_product=GeoFunc.crossProduct(vec1,vec2)
        Line1=LineString(line1)
        Line2=LineString(line2)
        inter={
            "length":0,
            "geom_type":None
        }
        # 只有平行才会有直线重叠
        if vec12_product==0:
            # copy避免影响原值
            new_line1=GeoFunc.copyPoly(line1)
            new_line2=GeoFunc.copyPoly(line2)
            if vec1[0]*vec2[0]<0 or vec1[1]*vec2[1]<0:
                new_line2=GeoFunc.reverseLine(new_line2)
            # 如果存在顶点相等，则选择其中一个
            if GeoFunc.almostEqual(new_line1[0],new_line2[0]) or GeoFunc.almostEqual(new_line1[1],new_line2[1]):
                inter["length"]=min(Line1.length,Line2.length)
                inter["geom_type"]='LineString'
                return inter
            # 排除只有顶点相交情况
            if GeoFunc.almostEqual(new_line1[0],new_line2[1]):
                inter["length"]=new_line2[1]
                inter["geom_type"]='Point'
                return inter
            if GeoFunc.almostEqual(new_line1[1],new_line2[0]):
                inter["length"]=new_line1[1]
                inter["geom_type"]='Point'
                return inter
            # 否则判断是否包含
            line1_contain_line2_pt0=GeoFunc.almostContain(new_line1,new_line2[0])
            line1_contain_line2_pt1=GeoFunc.almostContain(new_line1,new_line2[1])
            line2_contain_line1_pt0=GeoFunc.almostContain(new_line2,new_line1[0])
            line2_contain_line1_pt1=GeoFunc.almostContain(new_line2,new_line1[1])
            # Line1直接包含Line2
            if line1_contain_line2_pt0 and line1_contain_line2_pt1:
                inter["length"]=Line1.length
                inter["geom_type"]='LineString'
                return inter
            # Line2直接包含Line1
            if line1_contain_line2_pt0 and line1_contain_line2_pt1:
                inter["length"]=Line2.length
                inter["geom_type"]='LineString'
                return inter
            # 相互包含交点
            if line1_contain_line2_pt0 and line2_contain_line1_pt1:
                inter["length"]=LineString([line2[0],line1[1]]).length
                inter["geom_type"]='LineString'
                return inter
            if line1_contain_line2_pt1 and line2_contain_line1_pt0:
                inter["length"]=LineString([line2[1],line1[0]]).length
                inter["geom_type"]='LineString'
                return inter                
        return inter

    def reverseLine(line):
        pt0=line[0]
        pt1=line[1]
        return [[pt1[0],pt1[1]],[pt0[0],pt0[1]]]

    '''近似计算'''
    def almostEqual(point1,point2):
        if abs(point1[0]-point2[0])<bias and abs(point1[1]-point2[1])<bias:
            return True
        else:
            return False

    def extendLine(line):
        '''
        直线延长
        '''
        pt0=line[0]
        pt1=line[1]
        vect01=[pt1[0]-pt0[0],pt1[1]-pt0[1]]
        vect10=[-vect01[0],-vect01[1]]
        multi=40
        new_pt1=[pt0[0]+vect01[0]*multi,pt0[1]+vect01[1]*multi]
        new_pt0=[pt1[0]+vect10[0]*multi,pt1[1]+vect10[1]*multi]
        return [new_pt0,new_pt1]

    def getArc(line):
        if abs(line[0][0]-line[1][0])<0.01: # 垂直情况
            if line[0][1]-line[1][1]>0:
                return 0.5*math.pi
            else:
                return -0.5*math.pi
        k=(line[0][1]-line[1][1])/(line[0][0]-line[1][0])
        arc=np.arctan(k)
        return arc

    def extendInter(line1,line2):
        '''
        获得延长线的交点
        '''
        line1_extend=GeoFunc.extendLine(line1)
        line2_extend=GeoFunc.extendLine(line2)
        # 排查平行情况
        k1=GeoFunc.getArc(line1_extend)
        k2=GeoFunc.getArc(line2_extend)
        if abs(k1-k2)<0.01:
            return [line1[1][0],line1[1][1]]
        inter=mapping(LineString(line1_extend).intersection(LineString(line2_extend)))
        if inter["type"]=="GeometryCollection" or inter["type"]=="LineString":
            return [line1[1][0],line1[1][1]]
        return [inter["coordinates"][0],inter["coordinates"][1]]

    def twoDec(poly):
        for pt in poly:
            pt[0]=round(pt[0],2)
            pt[1]=round(pt[1],2)

    def similarPoly(poly):
        '''
        求解凸多边形的近似多边形，凹多边形内凹部分额外处理
        '''
        change_len=10
        extend_poly=poly+poly
        Poly=Polygon(poly)
        new_edges=[]
        # 计算直线平移
        for i in range(len(poly)):
            line=[extend_poly[i],extend_poly[i+1]]
            new_line=GeoFunc.slideOutLine(line,Poly,change_len)
            new_edges.append(new_line)
        
        # 计算直线延长线
        new_poly=[]
        new_edges.append(new_edges[0])
        for i in range(len(new_edges)-1):
            inter=GeoFunc.extendInter(new_edges[i],new_edges[i+1])
            new_poly.append(inter)
        
        GeoFunc.twoDec(new_poly) 

        return new_poly

    def slideOutLine(line,Poly,change_len):
        '''
        向外平移直线
        '''
        pt0=line[0]
        pt1=line[1]
        mid=[(pt0[0]+pt1[0])/2,(pt0[1]+pt1[1])/2]
        if pt0[1]!=pt1[1]:
            k=-(pt0[0]-pt1[0])/(pt0[1]-pt1[1]) # 垂直直线情况
            theta=math.atan(k)
            delta_x=1*math.cos(theta)
            delta_y=1*math.sin(theta)
            if Poly.contains(Point([mid[0]+delta_x,mid[1]+delta_y])):
                delta_x=-delta_x
                delta_y=-delta_y
            new_line=[[pt0[0]+change_len*delta_x,pt0[1]+change_len*delta_y],[pt1[0]+change_len*delta_x,pt1[1]+change_len*delta_y]]
            return new_line
        else:
            delta_y=1
            if Poly.contains(Point([mid[0],mid[1]+delta_y])):
                delta_y=-delta_y
            return [[pt0[0],pt0[1]+change_len*delta_y],[pt1[0],pt1[1]+change_len*delta_y]]

    def copyPoly(poly):
        new_poly=[]
        for pt in poly:
            new_poly.append([pt[0],pt[1]])
        return new_poly        

    def pointLineDistance(point, line):
        point_x = point[0]
        point_y = point[1]
        line_s_x = line[0][0]
        line_s_y = line[0][1]
        line_e_x = line[1][0]
        line_e_y = line[1][1]
        if line_e_x - line_s_x == 0:
            return abs(point_x - line_s_x),[line_s_x-point_x,0]
        if line_e_y - line_s_y == 0:
            return abs(point_y - line_s_y),[0,line_s_y-point_y]

        k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
        extend_line=[[point_x-1000,point_y-1000*(-1/k)],[point_x+1000,point_y+1000*(-1/k)]]
        inter=LineString(line).intersection(LineString(extend_line))
        if inter.is_empty==True:
            dis1=math.pow((point_x-line_s_x)*(point_x-line_s_x)+(point_y-line_s_y)*(point_y-line_s_y), 0.5)
            dis2=math.pow((point_x-line_e_x)*(point_x-line_e_x)+(point_y-line_e_y)*(point_y-line_e_y), 0.5)
            if dis1>dis2:
                return dis2,[line_e_x-point_x,line_e_y-point_y]
            else:
                return dis1,[line_s_x-point_x,line_s_y-point_y]
        else:
            pt=GeoFunc.getPt(inter)
            dis=math.pow((point_x-pt[0])*(point_x-pt[0])+(point_y-pt[1])*(point_y-pt[1]), 0.5)
            return dis,[pt[0]-point[0],pt[1]-point[1]]

    def getPt(point):
        mapping_result=mapping(point)
        return [mapping_result["coordinates"][0],mapping_result["coordinates"][1]]

    # 获得某个多边形的边
    def getPolyEdges(poly):
        edges=[]
        for index,point in enumerate(poly):
            if index < len(poly)-1:
                edges.append([poly[index],poly[index+1]])
            else:
                edges.append([poly[index],poly[0]])
        return edges

    def pointPrecisionChange(pt,num):
        return [round(pt[0],num),round(pt[1],num)]
    
    def linePrecisionChange(line,num):
        return [GeoFunc.pointPrecisionChange(line[0],num),GeoFunc.pointPrecisionChange(line[1],num)]
    
    def lineToVec(edge):
        return [edge[1][0]-edge[0][0],edge[1][1]-edge[0][1]]
    
    '''可能需要用近似计算进行封装！！！！！！'''
    def judgePosition(edge1,edge2):
        x1=edge1[1][0]-edge1[0][0]
        y1=edge1[1][1]-edge1[0][1]
        x2=edge2[1][0]-edge2[0][0]
        y2=edge2[1][1]-edge2[0][1]
        res=x1*y2-x2*y1
        right=False
        left=False
        parallel=False
        # print("res:",res)
        if res==0:
            parallel=True
        elif res>0:
            left=True
        else:
            right=True 
        return right,left,parallel


    def getSlideLine(line,x,y):
        new_line=[]
        for pt in line:
            new_line.append([pt[0]+x,pt[1]+y])
        return new_line

    def getCentroid(poly):
        return GeoFunc.getPt(Polygon(poly).centroid)
        
class PltFunc(object):

    def addPolygon(poly):
        for i in range(0,len(poly)):
            if i == len(poly)-1:
                PltFunc.addLine([poly[i],poly[0]])
            else:
                PltFunc.addLine([poly[i],poly[i+1]])

    def addPolygonColor(poly):
        for i in range(0,len(poly)):
            if i == len(poly)-1:
                PltFunc.addLine([poly[i],poly[0]],color="blue")
            else:
                PltFunc.addLine([poly[i],poly[i+1]],color="blue")

    def addLine(line,**kw):
        if len(kw)==0:
            plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],color="black",linewidth=0.5)
        else:
            plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],color=kw["color"],linewidth=0.5)            
    
    def showPlt(**kw):
        if len(kw)>0:
            if "minus" in kw:
                plt.axhline(y=0,c="blue")
                plt.axvline(x=0,c="blue")
                plt.axis([-kw["minus"],kw["width"],-kw["minus"],kw["height"]])
                
            else:
                plt.axis([0,kw["width"],0,kw["height"]])
        else:
            plt.axis([0,1000,0,1000])
            # plt.axis([-1000,2000,-979400.4498015114,20000])
            # plt.axis([-500,1000,0,1500])
        plt.show()
        plt.clf()

    def showPolys(polys):
        for poly in polys:
            PltFunc.addPolygon(poly)
        PltFunc.showPlt(width=2000,height=2000)

    def saveFig(name):
        plt.savefig('figs\\'+name+'.png')
        plt.cla()
    
    
class RatotionPoly():
    def __init__(self,angle):
        self.angle=angle
        self._max=360/angle

    def rotation(self,poly):
        if self._max>1:
            # print("旋转图形")
            rotation_res=random.randint(1,self._max-1)
            Poly=Polygon(poly)
            new_Poly=affinity.rotate(Poly,rotation_res*self.angle)
            mapping_res=mapping(new_Poly)
            new_poly=mapping_res["coordinates"][0]
            for index in range(0,len(poly)):
                poly[index]=[new_poly[index][0],new_poly[index][1]]
        else:
            pass
            # print("不允许旋转")

    def rotation_specific(self, poly, angle = -1):
        '''
        旋转特定角度
        '''
        Poly=Polygon(poly)
        if angle == -1: 
            angle = self.angle
        elif len(angle)>0:
            angle=np.random.choice(angle)
            # print('旋转{}°'.format(angle))
        new_Poly=affinity.rotate(Poly,angle)
        mapping_res=mapping(new_Poly)
        new_poly=mapping_res["coordinates"][0]
        for index in range(0,len(poly)):
            poly[index]=[new_poly[index][0],new_poly[index][1]]

class NFPArc(object):
    '''
    基于Arc的NFP计算
    参考文献：Irregular Packing Using the Line and Arc No-Fit Polygon(Burke et al. 2010)
    '''
    def __init__(self):
        pass

class NFPMinkowski(object):
    '''
    采用Minkowski Sum计算No fit polygon
    参考文献：The irregular cutting-stock problem a new procedure for deriving the no-fit polygon
    '''
    def __init__(self):
        pass

class NFP(object):
    def __init__(self,poly1,poly2,**kw):
        self.stationary=copy.deepcopy(poly1)
        self.sliding=copy.deepcopy(poly2)
        start_point_index=GeoFunc.checkBottom(self.stationary)
        self.start_point=[poly1[start_point_index][0],poly1[start_point_index][1]]
        self.locus_index=GeoFunc.checkTop(self.sliding)
        # 如果不加list则original_top是指针
        self.original_top=list(self.sliding[self.locus_index])
        GeoFunc.slideToPoint(self.sliding,self.sliding[self.locus_index],self.start_point)
        self.start=True # 判断是否初始
        self.nfp=[]
        self.rectangle=False
        if 'rectangle' in kw:
            if kw["rectangle"]==True:
                self.rectangle=True
        self.error=1
        self.main()
        if 'show' in kw:
            if kw["show"]==True:
                self.showResult()

    def main(self):
        i=0
        if self.rectangle: # 若矩形则直接快速运算 点的index为左下角开始逆时针旋转
            width=self.sliding[1][0]-self.sliding[0][0]
            height=self.sliding[3][1]-self.sliding[0][1]
            self.nfp.append([self.stationary[0][0],self.stationary[0][1]])
            self.nfp.append([self.stationary[1][0]+width,self.stationary[1][1]])
            self.nfp.append([self.stationary[2][0]+width,self.stationary[2][1]+height])
            self.nfp.append([self.stationary[3][0],self.stationary[3][1]+height])
        else:
            while self.judgeEnd()==False and i<75: # 大于等于75会自动退出的，一般情况是计算出错
            # while i<7:
                # print("########第",i,"轮##########")
                touching_edges=self.detectTouching()
                all_vectors=self.potentialVector(touching_edges)
                if len(all_vectors)==0:
                    print("没有可行向量")
                    self.error=-2 # 没有可行向量
                    break

                vector=self.feasibleVector(all_vectors,touching_edges)
                if vector==[]:
                    print("没有计算出可行向量")
                    self.error=-5 # 没有计算出可行向量
                    break
                
                self.trimVector(vector)
                if vector==[0,0]:
                    print("未进行移动")
                    self.error=-3 # 未进行移动
                    break

                GeoFunc.slidePoly(self.sliding,vector[0],vector[1])
                self.nfp.append([self.sliding[self.locus_index][0],self.sliding[self.locus_index][1]])
                i=i+1
                inter=Polygon(self.sliding).intersection(Polygon(self.stationary))
                if GeoFunc.computeInterArea(inter)>1:
                    print("出现相交区域")
                    self.error=-4 # 出现相交区域
                    break                

        if i==75:
            print("超出计算次数")
            self.error=-1 # 超出计算次数
    
    # 检测相互的连接情况
    def detectTouching(self):
        touch_edges=[]
        stationary_edges,sliding_edges=self.getAllEdges()
        # print(stationary_edges)
        # print(sliding_edges)
        for edge1 in stationary_edges:
            for edge2 in sliding_edges:
                inter=GeoFunc.intersection(edge1,edge2)
                if inter!=[]:
                    # print("edge1:",edge1)
                    # print("edge2:",edge2)
                    # print("inter:",inter)
                    # print("")
                    pt=[inter[0],inter[1]] # 交叉点
                    edge1_bound=(GeoFunc.almostEqual(edge1[0],pt) or GeoFunc.almostEqual(edge1[1],pt)) # 是否为边界
                    edge2_bound=(GeoFunc.almostEqual(edge2[0],pt) or GeoFunc.almostEqual(edge2[1],pt)) # 是否为边界
                    stationary_start=GeoFunc.almostEqual(edge1[0],pt) # 是否开始
                    orbiting_start=GeoFunc.almostEqual(edge2[0],pt) # 是否开始
                    touch_edges.append({
                        "edge1":edge1,
                        "edge2":edge2,
                        "vector1":self.edgeToVector(edge1),
                        "vector2":self.edgeToVector(edge2),
                        "edge1_bound":edge1_bound,
                        "edge2_bound":edge2_bound,
                        "stationary_start":stationary_start,
                        "orbiting_start":orbiting_start,
                        "pt":[inter[0],inter[1]],
                        "type":0
                    })
        return touch_edges 

    # 获得潜在的可转移向量
    def potentialVector(self,touching_edges):
        all_vectors=[]
        for touching in touching_edges:
            # print("touching:",touching)
            aim_edge=[]
            # 情况1
            if touching["edge1_bound"]==True and touching["edge2_bound"]==True:
                right,left,parallel=GeoFunc.judgePosition(touching["edge1"],touching["edge2"])
                # print("right,left,parallel:",right,left,parallel)
                if touching["stationary_start"]==True and touching["orbiting_start"]==True:
                    touching["type"]=0
                    if left==True:
                        aim_edge=[touching["edge2"][1],touching["edge2"][0]] # 反方向
                    if right==True:
                        aim_edge=touching["edge1"]
                if touching["stationary_start"]==True and touching["orbiting_start"]==False:
                    touching["type"]=1
                    if left==True:
                        aim_edge=touching["edge1"]
                if touching["stationary_start"]==False and touching["orbiting_start"]==True:
                    touching["type"]=2
                    if right==True:
                        aim_edge=[touching["edge2"][1],touching["edge2"][0]] # 反方向
                if touching["stationary_start"]==False and touching["orbiting_start"]==False:
                    touching["type"]=3
    
            # 情况2
            if touching["edge1_bound"]==False and touching["edge2_bound"]==True:
                aim_edge=[touching["pt"],touching["edge1"][1]]
                touching["type"]=4
            
            # 情况3
            if touching["edge1_bound"]==True and touching["edge2_bound"]==False:
                aim_edge=[touching["edge2"][1],touching["pt"]]
                touching["type"]=5

            if aim_edge!=[]:
                vector=self.edgeToVector(aim_edge)
                if self.detectExisting(all_vectors,vector)==False: # 删除重复的向量降低计算复杂度
                    all_vectors.append(vector)
        return all_vectors
    
    def detectExisting(self,vectors,judge_vector):
        for vector in vectors:
            if GeoFunc.almostEqual(vector,judge_vector):
                return True
        return False
    
    def edgeToVector(self,edge):
        return [edge[1][0]-edge[0][0],edge[1][1]-edge[0][1]]
    
    # 选择可行向量
    def feasibleVector(self,all_vectors,touching_edges):
        '''
        该段代码需要重构，过于复杂
        '''
        res_vector=[]
        # print("\nall_vectors:",all_vectors)
        for vector in all_vectors:
            feasible=True
            # print("\nvector:",vector,"\n")
            for touching in touching_edges:
                vector1=[]
                vector2=[]
                # 判断方向并进行转向
                if touching["stationary_start"]==True:
                    vector1=touching["vector1"]
                else:
                    vector1=[-touching["vector1"][0],-touching["vector1"][1]]
                if touching["orbiting_start"]==True:
                    vector2=touching["vector2"]
                else:
                    vector2=[-touching["vector2"][0],-touching["vector2"][1]]
                vector12_product=GeoFunc.crossProduct(vector1,vector2) # 叉积，大于0在左侧，小于0在右侧，等于0平行
                vector_vector1_product=GeoFunc.crossProduct(vector1,vector) # 叉积，大于0在左侧，小于0在右侧，等于0平行
                vector_vector2_product=GeoFunc.crossProduct(vector2,vector) # 叉积，大于0在左侧，小于0在右侧，等于0平行
                # print("vector:",vector)
                # print("type:",touching["type"])
                # print("vector12_product:",vector12_product)
                # print("vector1:",vector1)
                # print("vector2:",vector2)
                # print("vector_vector1_product:",vector_vector1_product)
                # print("vector_vector2_product:",vector_vector2_product)
                # 最后两种情况
                if touching["type"]==4 and (vector_vector1_product*vector12_product)<0:
                    feasible=False
                if touching["type"]==5 and (vector_vector2_product*(-vector12_product))>0:
                    feasible=False
                # 正常的情况处理
                if vector12_product>0:
                    if vector_vector1_product<0 and vector_vector2_product<0:
                        feasible=False
                if vector12_product<0:
                    if vector_vector1_product>0 and vector_vector2_product>0:
                        feasible=False
                # 平行情况，需要用原值逐一判断
                if vector12_product==0:
                    inter=GeoFunc.newLineInter(touching["edge1"],touching["edge2"])
                    # print("inter['geom_type']:",inter["geom_type"])
                    # print(inter)
                    if inter["geom_type"]=="LineString":
                        if inter["length"]>0.01:
                            # 如果有相交，则需要在左侧
                            if (touching["orbiting_start"]==True and vector_vector2_product<0) or (touching["orbiting_start"]==False and vector_vector2_product>0):
                                feasible=False
                    else:
                        # 如果方向相同，且转化直线也平行，则其不能够取a的方向
                        if touching["orbiting_start"]==True != touching["stationary_start"]==False and vector_vector1_product==0:
                            if touching["vector1"][0]*vector[0]>0: # 即方向相同
                                feasible=False
            #     if feasible==False:
            #         print("feasible:",False)
            #     print("")                     
            # print("feasible:",feasible)
            # print("")
            if feasible==True:
                res_vector=vector
                break
        return res_vector
        
    # 削减过长的向量
    def trimVector(self,vector):
        stationary_edges,sliding_edges=self.getAllEdges()
        new_vectors=[]
        for pt in self.sliding:
            for edge in stationary_edges:
                line_vector=LineString([pt,[pt[0]+vector[0],pt[1]+vector[1]]])
                end_pt=[pt[0]+vector[0],pt[1]+vector[1]]
                line_polygon=LineString(edge)
                inter=line_vector.intersection(line_polygon)
                if inter.geom_type=="Point":
                    inter_mapping=mapping(inter)
                    inter_coor=inter_mapping["coordinates"]
                    # if (end_pt[0]!=inter_coor[0] or end_pt[1]!=inter_coor[1]) and (pt[0]!=inter_coor[0] or pt[1]!=inter_coor[1]):
                    if (abs(end_pt[0]-inter_coor[0])>0.01 or abs(end_pt[1]-inter_coor[1])>0.01) and (abs(pt[0]-inter_coor[0])>0.01 or abs(pt[1]-inter_coor[1])>0.01):
                        # print("start:",pt)
                        # print("end:",end_pt)
                        # print("inter:",inter)
                        # print("")
                        new_vectors.append([inter_coor[0]-pt[0],inter_coor[1]-pt[1]])

        for pt in self.stationary:
            for edge in sliding_edges:
                line_vector=LineString([pt,[pt[0]-vector[0],pt[1]-vector[1]]])
                end_pt=[pt[0]-vector[0],pt[1]-vector[1]]
                line_polygon=LineString(edge)
                inter=line_vector.intersection(line_polygon)
                if inter.geom_type=="Point":
                    inter_mapping=mapping(inter)
                    inter_coor=inter_mapping["coordinates"]
                    # if (end_pt[0]!=inter_coor[0] or end_pt[1]!=inter_coor[1]) and (pt[0]!=inter_coor[0] or pt[1]!=inter_coor[1]):
                    if (abs(end_pt[0]-inter_coor[0])>0.01 or abs(end_pt[1]-inter_coor[1])>0.01) and (abs(pt[0]-inter_coor[0])>0.01 or abs(pt[1]-inter_coor[1])>0.01):
                        # print("start:",pt)
                        # print("end:",end_pt)
                        # print("inter:",inter)
                        # print("")
                        new_vectors.append([pt[0]-inter_coor[0],pt[1]-inter_coor[1]])
        
        # print(new_vectors)
        for vec in new_vectors:
            if abs(vec[0])<abs(vector[0]) or abs(vec[1])<abs(vector[1]):
                # print(vec)
                vector[0]=vec[0]
                vector[1]=vec[1]
    
    # 获得两个多边形全部边
    def getAllEdges(self):
        return GeoFunc.getPolyEdges(self.stationary),GeoFunc.getPolyEdges(self.sliding)
    
    # 判断是否结束
    def judgeEnd(self):
        sliding_locus=self.sliding[self.locus_index]
        main_bt=self.start_point
        if abs(sliding_locus[0]-main_bt[0])<0.1 and abs(sliding_locus[1]-main_bt[1])<0.1:
            if self.start==True:
                self.start=False
                # print("判断是否结束：否")
                return False
            else:
                # print("判断是否结束：是")
                return True
        else:
            # print("判断是否结束：否")
            return False

    # 显示最终结果
    def showResult(self):
        PltFunc.addPolygon(self.sliding)
        PltFunc.addPolygon(self.stationary)
        PltFunc.addPolygonColor(self.nfp)
        PltFunc.showPlt()

    # 计算渗透深度
    def getDepth(self):
        '''
        计算poly2的checkTop到NFP的距离
        Source: https://stackoverflow.com/questions/36972537/distance-from-point-to-polygon-when-inside
        '''
        d1=Polygon(self.nfp).distance(Point(self.original_top))
        # if point in inside polygon, d1=0
        # d2: distance from the point to nearest boundary
        if d1==0:
            d2=Polygon(self.nfp).boundary.distance(Point(self.original_top))
            # print('d2:',d2)
            return d2
        else: 
            return 0

# 计算NFP然后寻找最合适位置
def tryNFP():
    # df = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/euro_data/blaz.csv")
    # df = pd.read_csv("/Users/sean/Documents/Projects/Data/Shapes/now_fail.csv")

    # i=random.randint(0,80)
    # i=19
    # print(i)
    # poly1=json.loads(df['poly1'][i])
    # poly2=json.loads(df['poly2'][i])
    # GeoFunc.normData(poly1,50)
    # GeoFunc.normData(poly2,50)
    # GeoFunc.slidePoly(poly1,500,500)
    rect1=[[100,100], [150,100], [150,150], [100,150]]
    rect2=[[100,100], [120,100], [120,160], [100,160]]
    nfp=NFP(rect1,rect2,show=True,rectangle=True)
    print(nfp.nfp)
    # bfp=bestFitPosition(nfp,True)
    # print("Final fitness:",bfp.fitness)

def polygonFuncCheck():
    polygons=[[[525.0867175941048, 1460.9905830608504], [582.2017614687206, 1767.4083603084414], [377.1811708941839, 1756.4329089645125], [18.668940346513637, 1718.1056269354658], [125.08671759410481, 1460.9905830608504], [43.178931358332306, 1358.3845400631499], [175.0867175941048, 1201.1829619255186], [525.0867175941046, 1114.5804215470746], [584.8943387294362, 1310.99058306085]], [[882.6133890786109, 486.2860886342353], [566.730028708906, 507.71817542357644], [348.67234627062, 673.1201401481796], [327.2402594812786, 357.2367797784748], [342.03206833715944, 225.9558451637006], [522.2258419176433, 112.732592987212], [771.6217626611365, 0.0]], [[667.3308799144743, 1991.986091838081], [840.5359606713623, 2291.986091838081], [407.52325877914285, 2341.986091838081]], [[568.0825141100973, 921.3885247870273], [572.6421395707833, 1091.7668857250185], [404.64688704185494, 1063.006301794928], [333.48946790521234, 1161.9884459830212], [39.682506254725354, 1081.2983404009804], [178.2196310110113, 893.5049810733983], [207.93321282096156, 756.9138499276219], [268.05575589607827, 696.7913068525052], [376.18391938719776, 665.0420134187414], [484.31208287831726, 696.7913068525049], [712.6853325196702, 648.7856864365024]], [[1257.6976336467092, 1332.7108050946526], [1099.1256911845553, 1463.1287569092483], [964.6473071381935, 1666.8706587222596], [861.1196890971851, 1680.5003282066323], [617.615117493697, 1597.8416643231267], [667.9345238393714, 1332.2641372271364], [563.6862306850419, 1241.342470540617], [578.2769766225659, 997.6576157320133], [711.1196890971848, 1020.6927070713009], [861.1196890971848, 880.5003282066323], [1011.1196890971847, 1020.6927070713004], [1207.5298506109602, 1080.5003282066318]], [[1500.0, 613.9330217804386], [1250.0, 873.74064291577], [1100.0, 913.9330217804386], [793.5822227524088, 871.0480656550544], [700.0, 613.9330217804386], [818.0922137642275, 511.326978782738], [999.9999999999999, 440.7279410235508], [1152.0944533000788, 318.49069587677604], [1381.9077862357726, 511.32697878273757]], [[1500.0, 1955.8480328821652], [1296.458220183585, 2182.5729051884427], [1156.925935309314, 2351.7766096345385], [933.8339947992454, 2319.7008310239726], [797.7001702582966, 2217.7923264602796], [899.9999999999998, 1955.8480328821652], [918.0736009290961, 1872.7650302817883], [937.8077547633206, 1703.471973032811], [1043.0740646906856, 1559.9194561297923], [1361.9442935781137, 1653.5482031404617], [1483.7971894457992, 1843.1550101455932]], [[513.8181010090098, 2106.632917588077], [441.1638482084736, 2206.6329175880765], [416.3118960624631, 2330.146473601633], [230.9016994374947, 2330.1464736016333], [0.0, 2279.9436196300762], [38.28984286143282, 1952.124420400603], [111.47476339401464, 1832.697484357123], [276.6764582379096, 1748.5230165345458], [416.3118960624631, 1759.512563824541], [485.4101966249682, 1927.2724682545925]], [[1206.4850509928829, 251.76380902050414], [840.0948049781816, 300.00000000000045], [820.1147204772557, 148.23619097949586], [1359.7100472488446, 0.0]]]
    for poly in polygons:
        PltFunc.addPolygon(poly)
    PltFunc.showPlt(width=2500,height=2500)    

def getData(index=6):
    '''报错数据集有（空心）：han,jakobs1,jakobs2 '''
    '''形状过多暂时未处理：shapes、shirt、swim、trousers'''
    name=["ga","albano","blaz1","blaz2","dighe1","dighe2","fu","han","jakobs1","jakobs2","mao","marques","shapes","shirts","swim","trousers","convex","simple","ali2","ali3"]
    print("开始处理",name[index],"数据集")
    '''暂时没有考虑宽度，全部缩放来表示'''
    scale=[100,0.5,100,100,20,20,20,10,20,20,0.5,20,50,1,1,1,1,1,3,1,1,1,1,1]
    print("缩放",scale[index],"倍")
    user_name=os.getlogin()
    if user_name=='Prinway' or user_name=='mac':
        df = pd.read_csv("data/"+name[index]+".csv")
    else:
        df = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/data/"+name[index]+".csv")
    polygons=[]
    for i in range(0,df.shape[0]):
    # for i in range(0,4):
        for j in range(0,df['num'][i]):
            poly=json.loads(df['polygon'][i])
            GeoFunc.normData(poly,scale[index])
            polygons.append(poly)
    return polygons

def getConvex(**kw):
    if os.getlogin()=='Prinway':
        df = pd.read_csv("record/convex.csv")
    else:
        df = pd.read_csv("/Users/sean/Documents/Projects/data/convex.csv")
    polygons=[]
    poly_index=[]
    if 'num' in kw:
        for i in range(kw["num"]):
            poly_index.append(random.randint(0,7000))
    elif 'certain' in kw:
        poly_index=[1000,2000,3000,4000,5000,6000,7000]
    else:
        poly_index=[1000,2000,3000,4000,5000,6000,7000]
    # poly_index=[5579, 2745, 80, 6098, 3073, 8897, 4871, 4266, 3477, 3266, 8016, 4563, 1028, 10842, 1410, 7254, 5953, 82, 1715, 300]
    for i in poly_index:
        poly=json.loads(df['polygon'][i])
        polygons.append(poly)
    if 'with_index' in kw:
        return poly_index,polygons
    return polygons

if __name__ == '__main__':
    # tryNFP()
    # polygonFuncCheck()
    # PltFunc.addPolygonColor(((0, 580), (480, 580), (480, 200), (0, 200), (0, 580)))
    # PltFunc.addPolygon(((248.47, 860), (448.47, 940), (648.47, 940), (648.47, 560), (248.47, 560)))

    # PltFunc.addPolygon(((604.326, 180), (200, 180), (200, 760), (604.326, 760), (604.326, 180)))
    # PltFunc.addPolygonColor([[234.286,560],[360,560],[380,560],[380,723.959],[380,723.959],[380,460],[234.286,460],[234.286,560]])
    # PltFunc.addPolygon([[-80,580,],[200,580,],[200,400,],[-80,400,]])
    # PltFunc.addPolygon(((480, 200), (480, 380), (200, 380), (200, 760), (1e+08, 760), (1e+08, 200), (480, 200)))
    PltFunc.showPlt()
