'''
    This file is used for vectorization
'''
import numpy as np
import math
import json
from collections import Counter
import pyclipper 
from shapely.wkt import loads as load_wkt
from shapely.geometry import Polygon,LineString
from shapely.geometry import Point,mapping

# 量化算法部分
class vectorFunc(object):
    def __init__(self,polygon_vertexs,**kw):
        print("初始化处理器：",polygon_vertexs)
        self.error=1
        self.polygon_vertexs=polygon_vertexs
        self.centroid_out_force=False
        self.updateFeature()
        if self.error==-1:
            return
        self.cut_lines=[] # 切割直线
        self.origin_intersection=[] # 原始的交点
        self.vector=[] # 量化后的向量
        self.envelope_polygon=self.polygon_vertexs # 包络凸多边形
        if len(kw)>0:
            self.cut_nums=kw["cut_nums"]
        else:
            self.cut_nums=256

        self.vectorization()

    def vectorization(self):
        '''
        量化算法，图示参考readme
        1. centroid在内部：直接计算360度范围上的割线的交点，选择最外面的交点
        2. centroid在外部：选择外接矩形的边上的点，180度范围上的两个交点
        '''
        self.moveToStandard()
        print(self.polygon_vertexs)
        if self.error==-1:
            return
        self.computeMultiLines(self.cut_nums)
        self.vector=[]
        print("self.centroid_in",self.centroid_in)
        # 如果在内部
        if self.centroid_in and self.centroid_out_force==False:
            for index,line in enumerate(self.cut_lines):
                intersection=self.computeIntersection(line)
                if intersection!=None:
                    self.origin_intersection.append(intersection)
            for inter in self.origin_intersection:
                self.vector.append(float(format(self.distance(inter),'.2f')))
        else:
            all_intersection=[]
            for index,line in enumerate(self.cut_lines):
                intersection=self.outIntersection(line)
                all_intersection.append(intersection)
            
            start_nums=self.cut_nums*0.25*(self.out_id+1)
            all_inter_extend=all_intersection+all_intersection
            mid_inter=[]
            for i in range(int(start_nums),int(start_nums+self.cut_nums*0.5)):
                if all_inter_extend[i]["type"]=="LineString": # 一个交线
                    mid_inter.append(all_inter_extend[i]["coordinates"])
                elif all_inter_extend[i]["type"]=="MultiLineString": # 多个交线
                    mid_inter.append([all_inter_extend[i]["coordinates"][0][0],all_inter_extend[i]["coordinates"][1][1]])
                elif all_inter_extend[i]["type"]=="GeometryCollection":
                    # 如果是空集
                    if all_inter_extend[i]["geometries"]==[]:
                        mid_inter.append([])
                    else:
                        # 如果是多个，就选择直线返回
                        for _arr in all_inter_extend[i]["geometries"]:
                            if _arr["type"]=="LineString":
                                mid_inter.append(_arr["coordinates"])

            # 根据交点情况处理vector
            # 如果没有交点是9999999
            merge_vector=[[],[],[],[]]
            for index,inter in enumerate(mid_inter):
                if index<0.5*len(mid_inter):
                    if len(inter)==0:
                        merge_vector[0].append(9999999)
                        merge_vector[3].append(9999999)
                    else:
                        merge_vector[0].append(float(format(self.distance(inter[0]),'.2f')))
                        merge_vector[3].append(-1*float(format(self.distance(inter[1]),'.2f')))
                else:
                    if len(inter)==0:
                        merge_vector[1].append(9999999)
                        merge_vector[2].append(9999999)
                    else:
                        merge_vector[1].append(float(format(self.distance(inter[0]),'.2f')))
                        merge_vector[2].append(-1*float(format(self.distance(inter[1]),'.2f')))

            # 向量前后转向
            merge_vector[2]=self.arrInverse(merge_vector[2])
            merge_vector[3]=self.arrInverse(merge_vector[3])

            # 合并向量
            if self.out_id==0:
                self.vector=merge_vector[3]+merge_vector[0]+merge_vector[1]+merge_vector[2]
            elif self.out_id==1:
                self.vector=merge_vector[2]+merge_vector[3]+merge_vector[0]+merge_vector[1]
            elif self.out_id==2:
                self.vector=merge_vector[1]+merge_vector[2]+merge_vector[3]+merge_vector[0]
            else:
                self.vector=merge_vector[0]+merge_vector[1]+merge_vector[2]+merge_vector[3]

        print('vector:',self.vector)


    def centOutUpdate(self):
        '''
        centroid在外部的时候，选择一个外接矩形的边的中点
        '''
        polygon = Polygon(self.polygon_vertexs)
        # 几个参考值
        x_min=polygon.bounds[0]
        y_min=polygon.bounds[1]
        x_max=polygon.bounds[2]
        y_max=polygon.bounds[3]
        x_mean=(x_min+x_max)*0.5
        y_mean=(y_min+y_max)*0.5
        x_quan0=x_min*0.25+x_max*0.75
        y_quan0=y_min*0.25+y_max*0.75
        x_quan1=x_max*0.25+x_min*0.75
        y_quan1=y_max*0.25+y_min*0.75

        # 四个点
        cen_pts=[[x_max+2,y_mean],[x_mean,y_min-2],[x_min-2,y_mean],[x_mean,y_max+2]]
        end_pts=[[x_max+2,y_max+2],[x_max+2,y_min-2],[x_min-2,y_min-2],[x_min-2,y_max+2]]
        cen_pts_extend=cen_pts+cen_pts
        end_pts_extend=end_pts+end_pts

        # 四个中间点的判断，符合条件就更新
        for index,pt in enumerate(cen_pts):
            lines=[[pt,cen_pts_extend[index+2]],[pt,end_pts_extend[index+2]],[pt,end_pts_extend[index+3]]]
            is_key=True
            for line in lines:
                inter=mapping(LineString(line).intersection(polygon))
                if inter["type"]=="LineString":
                    pass
                elif inter["type"]=="GeometryCollection":
                    if inter["geometries"]==[]:
                        return []
                    else:
                        is_key=False
                else:
                    is_key=False
            if is_key==True:
                self.out_id=index
                return pt

        self.error=-1
        print("############外部交点错误############")
    
    # 顶点值过小扩大50倍
    def standardEdges(self,polygon):
        '''
        数值计算问题，部分数据的边太小了近似计算会出错
        如果顶点的值小于70，全部*50
        '''
        # 更新最低值
        x_min=0
        y_min=0
        if polygon.bounds[0]<0:
            x_min=-polygon.bounds[0]
        if polygon.bounds[1]<0:
            y_min=-polygon.bounds[1]

        for point in self.polygon_vertexs:
            point[0]=point[0]+x_min
            point[1]=point[1]+y_min

        # 部分数据的值过低
        if polygon.bounds[2]<70:
            for point in self.polygon_vertexs:
                point[0]=point[0]*50
                point[1]=point[1]*50
        
        self.updateFeature()
        
    def moveToStandard(self):
        '''
        数据标准化，然后多边形重心平移到x=y的位置
        如果在centroid外部就是key point平移到x=y，并将坐标平移
        '''
        polygon = Polygon(self.polygon_vertexs)
        self.standardEdges(polygon)

        # 判断更新x还是更新y，centroid平移到中心
        coord_choose=1
        difference=self.centroid_x-self.centroid_y

        if self.centroid_y>self.centroid_x:
            coord_choose=0
            difference=difference*(-1)
            
        for point in self.polygon_vertexs:
            point[coord_choose]=point[coord_choose]+difference
        
        # print("self.centroid_in",self.centroid_in)
        if self.centroid_in==False:
            polygon = Polygon(self.polygon_vertexs)
            bounds=[polygon.bounds[0],polygon.bounds[1],polygon.bounds[2],polygon.bounds[3]]
            for ver in self.polygon_vertexs:
                ver[0]=ver[0]+bounds[2]-bounds[0]
                ver[1]=ver[1]+bounds[3]-bounds[1]

        self.updateFeature()
            
    def updateFeature(self):
        '''
        根据多边形情况更新参数
        '''
        self.polygon=Polygon(self.polygon_vertexs)

        self.centroid_x=self.polygon.centroid.xy[0][0]
        self.centroid_y=self.polygon.centroid.xy[1][0]
        # 补充更新
        self.centroid=Point(self.centroid_x,self.centroid_y)
        self.centroid_in= self.polygon.contains(self.centroid)

        # 重心在外部，则更新为centout情况下的cent
        if self.centroid_in==False or self.centroid_out_force==True:
            centroid=self.centOutUpdate()
            if self.error==-1:
                return
            self.centroid_x=centroid[0]
            self.centroid_y=centroid[1]
            self.centroid=Point(self.centroid_x,self.centroid_y)
            # print("更新centroid:",centroid)
    

    def computeMultiLines(self,num):
        '''
        计算目标的割线
        '''
        eighth_num=num/8
        half_num=num/2
        basic_angle=math.pi*2/num
        half_width=self.centroid_x

        for i in range(0,num):
            angle=basic_angle*i
            if i<eighth_num:
                delta_x=half_width
                delta_y=delta_x*math.tan(angle)
            elif eighth_num<=i<eighth_num*3:
                delta_y=half_width
                delta_x=delta_y/math.tan(angle)
            elif eighth_num*3<=i<eighth_num*5:
                delta_x=-half_width
                delta_y=delta_x*math.tan(angle)
            elif eighth_num*5<=i<eighth_num*7:
                delta_y=-half_width
                delta_x=delta_y/math.tan(angle)
            else:
                delta_x=half_width
                delta_y=delta_x*math.tan(angle)
            
            x=int(self.centroid_x+delta_x)
            y=int(self.centroid_y-delta_y)

            line=[[x,y], [self.centroid_x,self.centroid_y]]
            self.cut_lines.append(line)
    
    # 计算一个点到质心的距离    
    def distance(self,point):
        return math.sqrt(math.pow(point[0]-self.centroid_x,2)+math.pow(point[1]-self.centroid_y,2))    

    def computeIntersection(self,line):
        '''
        计算交点
        '''
        P=Polygon(self.polygon_vertexs)
        L=LineString(line)
        inter=mapping(P.intersection(L))
        if inter["type"]=="GeometryCollection":
            if inter["geometries"]==[]:
                return []
        elif inter["type"]=="LineString":
            return self.getFarest(inter)
        elif inter["type"]=="Point":
            return [inter["coordinates"][0],inter["coordinates"][1]]
        elif inter["type"]=="MultiLineString":
            return self.getFarest(inter)
        else:
            print("———————————此情况需要考虑————————理论上没有")
    
    def linePoly(self,line):
        P=Polygon(self.polygon_vertexs)
        L=LineString(line)
        inter=mapping(P.intersection(L))
        return inter

    def outIntersection(self,line):
        '''
        此时返回的是全部的交点
        '''
        P=Polygon(self.polygon_vertexs)
        L=LineString(line)
        inter=mapping(P.intersection(L))
        return inter
    
    def getFarest(self,inter):
        '''
        多个交点的情况下获得最远的交点
        '''
        _far_pt=[self.centroid_x,self.centroid_y]
        _far_dis=0
        centroid=[self.centroid_x,self.centroid_y]
        if inter["type"]=="MultiLineString":
            for line in inter["coordinates"]:
                for pt in line:
                    if self.distance(pt)>_far_dis:
                        _far_pt=pt
                        _far_dis=self.distance(pt)
            return _far_pt
        if inter["type"]=="LineString":
            for pt in inter["coordinates"]:
                if self.distance(pt)>_far_dis:
                    _far_pt=pt
                    _distance=self.distance(pt)
            return _far_pt
        

# 重建图像及其评价类
# 输入向量化结果和quantiFunc的类，评价向量化的效果
class rebuildEvalute(object):

    def __init__(self,vect,pp):
        self.vect=vect
        self.pp=pp
        self.vertexs=self.rebuildShape()
        self.normPosi()
        self.error=1
        self.judgeBuild()
    
    def judgeBuild(self):
        '''
        包含率和超出率评价rebuild效果
        '''
        rebuildP=Polygon(self.vertexs)
        originP=Polygon(self.pp.polygon_vertexs)
        if rebuildP.is_valid==False or originP.is_valid==False:
            print("#########重建错误，问题暂时无法解决#####")
            self.error=-1
            return
        interP=rebuildP.intersection(originP)
        self.inter_area=self.computeArea(mapping(interP))
        self.contain_ratio=self.inter_area/originP.area
        self.exceed_ratio=(rebuildP.area-self.inter_area)/rebuildP.area
        print("contain_ratio:",self.contain_ratio,",exceed_ratio:",self.exceed_ratio)

    def rebuildShape(self):
        '''
        通过vector重建多边形
        '''
        _len=len(self.vect)
        base=2*math.pi/_len
        vertexs=[]
        for index,dire in enumerate(self.vect):
            if dire==9999999:
                continue
            # 计算距离
            angle=index*base
            # 正负注意一下
            y = -dire*math.sin(angle)
            x = dire*math.cos(angle)
            if dire < 0:
                if self.pp.out_id==1 or self.pp.out_id==3:
                    x=-x
                else:
                    y=-y
            vertexs.append([x,y])
        return vertexs
    
    def normPosi(self):
        '''
        平移位置到origin的位置，后续可以计算相交区域
        '''
        for ver in self.vertexs:
            ver[0]=ver[0]+self.pp.centroid_x
            ver[1]=ver[1]+self.pp.centroid_y

    def computeArea(self,inter):
        '''
        计算相交区域的面积
        '''
        if inter["type"]=="Polygon":
            poly=inter["coordinates"][0]
            return Polygon(poly).area
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

if __name__ == '__main__':

    low=[[0.0, 0.0], [2.0, -1.0], [4.0, 0.0], [4.0, 3.0], [2.0, 4.0], [0.0, 3.0]]

    poly=low

    pp=vectorFunc(poly,cut_nums=16)
    rebuildEvalute(pp.vector,pp)
    
