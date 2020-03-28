'''
    该文件主要用于几何算法
'''
from cv2 import cv2
from shapely.geometry import Polygon,Point,mapping,LineString
from shapely.wkt import loads as load_wkt
from shapely.ops import unary_union
import pyclipper 
import math
import numpy as np
import pandas as pd
import json
import logging
import time
logging.basicConfig(level=logging.INFO,format="%(asctime)s line %(lineno)d %(levelname)s:%(message)s", datefmt="%H:%M")
logger = logging.getLogger(__name__)

class graphCV(object):
    def __init__(self):
        self.begin=1
        self.img = np.zeros((2000,2000,3),np.uint8)+255
    
    def addPolygon(self,new_polygon):
        pts=np.array(new_polygon,np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(self.img,[pts],True,(0,0,0)) 

    def addPolygonColor(self,new_polygon):
        pts=np.array(new_polygon,np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(self.img,[pts],True,(237,88,32)) 

    def addLine(self,line):
        pts=np.array(line,np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(self.img,[pts],False,(0,0,0)) 

    def addLineColor(self,line):
        pts=np.array(line,np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(self.img,[pts],False,(237,88,32)) 

    def showPolygon(self,**kw):
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image',1001,1001)#定义frame的大小
        cv2.imshow('image',self.img)
        if len(kw)>0:
            cv2.imwrite('/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/image/'+kw["_type"]+time.asctime(time.localtime(time.time()))+'.png', self.img)            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.destroyAllWindows()

class geoFunc(object):
    '''
    almostContain: 一条直线包含了一个点（可能出现近似计算问题）
    getPloyEdges: 获得一个Polygon的所有的边
    checkTop: 多边形的最高点
    checkBottom: 多边形的最低点
    computeInterArea: 计算相交区域的面积，分情况处理
    extendLine: 延长直线
    extendInter: 延长直线的焦点
    getIndex: 获得点在多边形上的index
    getSlide: 获得平移后的结果
    slidePoly: 直接平移多边形
    vectorLength: 获得向量的长度
    judgePointInVector: 判断点是否在向量上
    '''
    def almostContain(line,point):
        '''
        判断点是否在直线上
        '''
        pt1=[int(line[0][0]),int(line[0][1])]
        pt2=[int(line[1][0]),int(line[1][1])]
        point=[int(point[0]),int(point[1])]

        # 水平直线情况
        if abs(pt1[1]-point[1])<0.001 and abs(pt2[1]-point[1])<0.001:
            if (pt1[0]-point[0])*(pt2[0]-point[0])<0:     
                return True

        # 点在直线两端
        if pt1==point or pt2==point:
            return True
    
        # 排除垂直的情况
        if pt1[0]==pt2[0]:
            # 如果目标点也是垂直
            if point[0]==pt1[0]:
                if (point[1]-pt1[1])*(pt2[1]-point[1])>0:
                    return True
                else:
                    return False
            else:
                return False
                
        # 排除其他垂直
        if pt1[0]==point[0] or pt2[0]==point[0]:
            return False

        # y1-y2/x1-x2
        k1=(line[0][1]-line[1][1])/(line[0][0]-line[1][0])
        k2=(point[1]-line[1][1])/(point[0]-line[1][0])
        # 水平直线
        if k1==0 and k2==0 and (point[0]-line[1][0])*(line[0][0]-point[0])>0:
            return True
        if abs(k1-k2)<0.1:
            # 判断是否同一个方向
            if (point[1]-pt1[1])*(pt2[1]-point[1])>0:
                return True
            else:
                return False
        else:
            return False
    
    def getPloyEdges(poly):
        edges=[]
        for index,point in enumerate(poly):
            if index < len(poly)-1:
                if poly[index]!=poly[index+1]: # 排除情况
                    edges.append([poly[index],poly[index+1]])
            else:
                if poly[index]!=poly[0]: # 排除情况
                    edges.append([poly[index],poly[0]])
        return edges

    def checkTop(poly):
        polyP=Polygon(poly)
        min_y=polyP.bounds[1]
        for index,point in enumerate(poly):
            if point[1]==min_y:
                return index

    def checkBottom(poly):
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
        return geoFunc.checkLeft(poly), geoFunc.checkBottom(poly), geoFunc.checkRight(poly), geoFunc.checkTop(poly)
    
    def slideToPoint(poly,pt1,pt2):
        geoFunc.slidePoly(poly,pt2[0]-pt1[0],pt2[1]-pt1[1])

    def computeInterArea(inter):
        '''
        计算相交区域的面积
        '''
        # 一个多边形
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
        return 0

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

    def extendInter(line1,line2):
        '''
        获得延长线的交点
        '''
        line1_extend=geoFunc.extendLine(line1)
        line2_extend=geoFunc.extendLine(line2)
        inter=mapping(LineString(line1_extend).intersection(LineString(line2_extend)))
        if inter["type"]=="GeometryCollection" or inter["type"]=="LineString":
            return [line1[1][0],line1[1][1]]
        return [inter["coordinates"][0],inter["coordinates"][1]]

    def getSlide(poly,vector):
        '''
        获得平移后的情况
        '''
        new_vertex=[]
        for point in poly:
            new_point=[point[0]+vector[0],point[1]+vector[1]]
            new_vertex.append(new_point)
        return new_vertex

    def slidePoly(poly,x,y):
        for point in poly:
            point[0]=point[0]+x
            point[1]=point[1]+y

    def vectorLength(vector):
        return math.sqrt(math.pow(vector[0],2)+math.pow(vector[1],2))

    def judgePointInVector(point,vectors):
        belong=False
        for vector in vectors:
            if vector["start"]==point["point_cord"]:
                belong=True
                break
        return belong

    def polyToArr(inter):
        res=mapping(inter)
        _arr=[]
        if res["type"]=="MultiPolygon":
            for poly in res["coordinates"]:
                for point in poly[0]:
                    _arr.append([point[0],point[1]])
        else:
            for point in res["coordinates"][0]:
                _arr.append([point[0],point[1]])
        return _arr
    
    def unionPolyons(polygons):
        poly=Polygon(polygons[0])
        for i in range(1,len(polygons)):
            poly=poly.union(Polygon(polygons[i]))
        return geoFunc.polyToArr(poly)

    def normData(poly,num):
        for ver in poly:
            ver[0]=ver[0]*num
            ver[1]=ver[1]*num

class NFP(object):
    def __init__(self,main,adjoin,show):
        self.main=main
        self.adjoin=adjoin
        self.original_adjoin=adjoin
        self.main_extend=self.main+self.main
        self.adjoin_extend=self.adjoin+self.adjoin
        self.NFP=[]
        self.error=1
        self.show=show
        self.gCV=graphCV()
        self.run()        

    def run(self):
        self.iniPosi()
        i=0
        
        # 限制200个移动次数，否则是出错了
        while i<200:
            main_ad,ad_main=self.nfpIS() # 计算交点情况
            res=self.slideByInter(main_ad,ad_main) # 根据相交情况平移
            # 数据异常情况
            if res==-1:
                self.error=-1
                print("###################数据错误###################")
                return

            if self.judgeEnd():
                print("计算NFP结束")
                break
            i=i+1

        if i==200:
            print("###################数据超出###################")
            return 
        
        if self.show==True:
            self.showAll()

    # 显示排样后的结果
    def showAll(self):
        self.gCV.addPolygon(self.main)
        self.gCV.addPolygon(self.adjoin)
        self.gCV.addPolygonColor(self.NFP)
        self.gCV.showPolygon()

    def nfpIS(self):
        '''
        两个多边形的交点情况计算
        '''
        main_edges,adjoin_edges=self.getEdges()
        main_pts=self.main
        adjoin_pts=self.adjoin

        main_ad=[] # main edge to adjoin point

        # 判断是否存在直线的交点
        for i,point in enumerate(adjoin_pts):
            for j,edge in enumerate(main_edges):
                if geoFunc.almostContain(edge,point):
                    sect={
                        "point_cord":[point[0],point[1]],
                        "point_index":i,
                        "edge_index":[j],
                        "sect_type":0 # 0表示点在直线上，1表示相交于顶点
                    }
                    main_ad.append(sect)

        main_ad_judged=[]
        
        for inter in main_ad:
            status=0
            for judged in main_ad_judged:
                if judged["point_cord"]==inter["point_cord"]:
                    judged["sect_type"]=1
                    judged["edge_index"].append(inter["edge_index"][0])
                    status=1
                    break
            if status==0:
                main_ad_judged.append(inter)

        ad_main=[] # adjoin edge to main points
        for i,point in enumerate(main_pts):
            for j,edge in enumerate(adjoin_edges):
                if geoFunc.almostContain(edge,point):
                    sect={
                        "point_cord":[point[0],point[1]],
                        "point_index":i,
                        "edge_index":[j]
                    }
                    ad_main.append(sect)
        
        # 如果只有一个交点，设置为begin
        if len(main_ad_judged)==1:
            self.inter_begin=main_ad_judged[0]

        # 如果两个都没有交点，就返回判断
        return main_ad_judged,ad_main
    
    def slideByInter(self,main_ad,ad_main):
        '''
        根据交点情况进行平移
        '''
        ad_domin=self.adjoin[self.ad_domin_index]
        vectors=[]
        for inter in main_ad:
            vector=self.getByPt(inter)
            if vector==False:
                pass
            else:
                vectors.append({
                    "start":inter["point_cord"],
                    "vector":vector
                })

        # 首先需要判断不在上述的向量中 
        for inter in ad_main:
            if geoFunc.judgePointInVector(inter,vectors)==False:
                vector=self.getByPt_(inter)
                if vector==False:
                    pass
                else:
                    vectors.append({
                        "start":inter["point_cord"],
                        "vector":vector
                    })
        
        # 排除异常数据
        if len(vectors)==0:
            return -1

        _max={
            "len":geoFunc.vectorLength(vectors[0]["vector"]),
            "vector":vectors[0]["vector"]
        }
        self.last_vector=_max["vector"]
        # self.gCV.addLineColor([ad_domin,[ad_domin[0]+ _max["vector"][0],ad_domin[1]+ _max["vector"][1]]])
        
        self.slidePoly(self.adjoin, _max["vector"][0], _max["vector"][1])
        
        point=self.adjoin[self.ad_domin_index]

        self.NFP.append([int(point[0]),int(point[1])])
        return _max["vector"]

    def getByPt(self,inter):
        '''
        根据单个交点情况返回平移向量，adjoin的点在main上
        '''
        start_pt=self.adjoin_extend[inter["point_index"]]
        # 交点在直线上/顶尖相交
        if len(inter["edge_index"])>=2:
            next_pt_main=self.main_extend[self.getLarge(inter["edge_index"][0],inter["edge_index"][1],len(self.main))+1]
        else:
            next_pt_main=self.main_extend[inter["edge_index"][0]+1]

        next_pt_ad=self.adjoin_extend[inter["point_index"]+1]

        # 计算两个方向，不管怎么样肯定是逆时针的，也就是点+1
        vec_main=[next_pt_main[0]-start_pt[0],next_pt_main[1]-start_pt[1]]
        vec_ad=[(next_pt_ad[0]-start_pt[0])*-1,(next_pt_ad[1]-start_pt[1])*-1]
        
        # 判断该方向能否平移
        if self.judgeDirec(vec_main)!=1:
            vector=vec_main
        elif self.judgeDirec(vec_ad)!=1:
            vector=vec_ad
        else:
            return False
        
        return self.getFinalVector(vector)

    # 根据main的顶点在adjoin上的情况计算，情况更少
    def getByPt_(self,inter):
        start_pt=self.main_extend[inter["point_index"]]
        next_pt=self.adjoin_extend[inter["edge_index"][0]+1]
        vector=[(next_pt[0]-start_pt[0])*(-1),(next_pt[1]-start_pt[1])*(-1)]
        
        if self.judgeDirec(vector)==1:
            return False
        
        return self.getFinalVector(vector)
    
    def getFinalVector(self,vector):
        '''
        进行向量的裁剪，如果按照vector平移后会产生相交区域，则往回平移
        裁剪后继续判断是否最终结果
        '''
        while True:
            check=self.checkVector(vector)
            if check==vector:
                return vector
            else:
                vector=check

        return vector

    # 检查vector方向平移效果
    def checkVector(self,vector):
        '''
        检查按照某个vector平移是否会有交点，如果有交点，就计算一个距离更短的向量
        具体请参考：http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.440.379&rep=rep1&type=pdf
        '''
        origin=vector
        judge=geoFunc.getSlide(self.adjoin,vector)
        judgeP=Polygon(judge)
        
        res=mapping(judgeP.intersection(self.mainP))
        main_edges,adjoin_edges=self.getEdges()
        adjoinP=Polygon(self.adjoin)

        # 如果平移后相交区域大于2，即有相交区域，需要裁剪
        if self.judgeArea(res,2)==False:            
            for index,point in enumerate(self.adjoin+self.main):
                # 分别计算adjoin的正方向和main的反向
                if index<len(self.adjoin):
                    line=[point,[point[0]+vector[0],point[1]+vector[1]]]  
                    inter=mapping(self.mainP.intersection(LineString(line)))
                else:
                    line=[point,[point[0]-vector[0],point[1]-vector[1]]]
                    inter=mapping(adjoinP.intersection(LineString(line)))
                
                # 判断是否存在相交区域
                if inter["type"]=="GeometryCollection":
                    if inter["geometries"]==[]:
                        continue
                    else:
                        for detail in inter["geometries"]:
                            if detail["type"]=="LineString":
                                res=self.getCutVector(detail,point,index)
                                if res==False:
                                    continue
                                else:
                                    return res
                elif inter["type"]=="LineString":
                    res=self.getCutVector(inter,point,index)
                    if res==False:
                        continue
                    else:
                        return res

        return vector
    
    # 如果vector平移后有相交，cut的情况
    def getCutVector(self,inter,point,index):
        if self.judgeEdgeContain(inter["coordinates"])==False:
            return False
        mid=self.getAnotherPoint(inter["coordinates"],point)
        if index<len(self.adjoin):
            vector=[mid[0]-point[0],mid[1]-point[1]]
        else:
            vector=[point[0]-mid[0],point[1]-mid[1]]
        return vector

    # 判断某个方向是否可以平移
    def judgeDirec(self,vector):
        '''
        通过在某方向平移很小的距离，计算是否有相交区域，判断某个方向是否可行
        '''
        if vector==[0,0]:
            return 1

        # 参考向量，极小值
        vec_len=math.sqrt(math.pow(vector[0],2)+math.pow(vector[1],2))/2
        direc=[vector[0]/vec_len,vector[1]/vec_len]
        new_vertex=geoFunc.getSlide(self.adjoin,direc)
        # 设置多边形
        mainP=Polygon(self.main)
        newP=Polygon(new_vertex)
        # 交点情况
        inter=mapping(newP.intersection(mainP))

        if len(inter)==0:
            return 0 # 没有交点
        else:
            if inter["type"]=="Polygon" or inter["type"]=="MultiPolygon" or inter["type"]=="GeometryCollection":
                if self.judgeArea(inter,0.01)==True:
                    return 0
                else:
                    return 1
            else:
                return 2  # 点在直线

        return 0

    # 点在直线上，返回另一个顶点
    def getAnotherPoint(self,cord,point):
        if cord[0][0]==point[0] and cord[0][1]==point[1]:
            return [cord[1][0],cord[1][1]]
        else:
            return [cord[0][0],cord[0][1]]

    # 判断直线是否在main和adjoin的边界上
    def judgeEdgeContain(self,line):
        '''
        判断相交直线是否在ajoin和main中
        如果在的话，这就是应该平移的直线
        '''
        main,adjoin=self.getEdges()
        for poly in [main,adjoin]:
            for edge in poly:
                inter=mapping(LineString(edge).intersection(LineString(line)))
                if inter["type"]=="LineString":
                    return False
                if geoFunc.almostContain(line,edge[0])==True and geoFunc.almostContain(line,edge[1])==True:
                    return False
                if geoFunc.almostContain(edge,line[0])==True and geoFunc.almostContain(edge,line[1])==True:
                    return False
        return True
    
    # 判断交叉面积是否小于标准
    def judgeArea(self,inter,crit):
        if geoFunc.computeInterArea(inter)<crit:
            return True
        else:
            return False

    # 初始化位置
    def iniPosi(self):
        # 标准化位置
        # self.slidePoly(self.main,150,150)
        # 计算应该平移的值
        main_start=geoFunc.checkTop(self.main)
        adjoin_bot=geoFunc.checkBottom(self.adjoin)
        # 平移到旁边的位置
        diff_x=self.main[main_start][0]-self.adjoin[adjoin_bot][0]
        diff_y=self.main[main_start][1]-self.adjoin[adjoin_bot][1]
        self.slidePoly(self.adjoin,diff_x,diff_y)
        self.ad_domin_index=adjoin_bot
        self.main_start=main_start
        self.mainP=Polygon(self.main)
        self.NFP=[[int(self.adjoin[adjoin_bot][0]),int(self.adjoin[adjoin_bot][1])]]
    
    # 判断是否结束
    def judgeEnd(self):
        '''
        通过点来判断执行过程是否结束
        '''
        adjoin_bot=self.adjoin[self.ad_domin_index]
        main_pt=self.main[self.main_start]
        last_line=[[adjoin_bot[0]-self.last_vector[0],adjoin_bot[1]-self.last_vector[1]],adjoin_bot]
        if abs(adjoin_bot[0]-main_pt[0])<0.1 and abs(adjoin_bot[1]-main_pt[1])<0.1:
            return True
        elif geoFunc.almostContain(last_line,main_pt)==True and main_pt!=last_line[0]:
            return True
        else:
            return False

    # 平移多边形并判断是否
    def slidePoly(self,poly,x,y):
        '''
        平移样片并更新extend
        '''
        for point in poly:
            point[0]=point[0]+x
            point[1]=point[1]+y

        self.adjoin_extend=self.adjoin+self.adjoin
        self.main_extend=self.main+self.main
    
    # 获得main和adjoin的全部边
    def getEdges(self):
        '''
        获得main和adjoin的所有边
        '''
        # 存储进入直线合集
        main_edges=[]
        for index,point in enumerate(self.main):
            if index < len(self.main)-1:
                main_edges.append([self.main[index],self.main[index+1]])
            else:
                main_edges.append([self.main[index],self.main[0]])
        
        # 存储进入直线合集
        adjoin_edges=[]
        for index,point in enumerate(self.adjoin):
            if index < len(self.adjoin)-1:
                adjoin_edges.append([self.adjoin[index],self.adjoin[index+1]])
            else:
                adjoin_edges.append([self.adjoin[index],self.adjoin[0]])

        return main_edges,adjoin_edges
    
    def getLarge(self,arg1,arg2,max_len):
        _max=arg1
        _min=arg2
        if arg1 < arg2:
            _max=arg2
            _min=arg1

        if _max==max_len-1 and _min==0:
            return 0
        else:
            return _max


# 放置形状的函数
class PlacePolygons(object):
    def __init__(self):
        self.width=1000
        self.height=2000

    def placePolygons(self,polygons,show):
        self.polygons=polygons
        self.normData(50)
        self.gCV=graphCV()
        self.placeFirstPoly()

        for i in range(1,len(polygons)):
            self.placePoly(i)
        
        self.getHeight()
        print("height:",self.contain_height)
        
        if show==True:
            self.showAll()

    def getHeight(self):
        _max=0
        for i in range(1,len(self.polygons)):
            bottom_index=geoFunc.checkBottom(self.polygons[i])
            bottom=self.polygons[i][bottom_index][1]
            if bottom>_max:
                _max=bottom
        self.contain_height=_max
        self.gCV.addLineColor([[self.width,0],[self.width,self.contain_height]])
        self.gCV.addLineColor([[0,self.contain_height],[self.width,self.contain_height]])

    def placePoly(self,index):
        '''
        放置某一个index的形状进去
        '''
        adjoin=self.polygons[index]
        self.getInnerFitRectangleNew(self.polygons[index])
        ifp=self.getInnerFitRectangle(self.polygons[index])
        differ_region=Polygon(ifp)
        # 求解NFP和IFP的资料
        for main_index in range(0,index):
            main=self.polygons[main_index]
            nfp=NFP(main,adjoin,False).NFP
            differ_region=differ_region.difference(Polygon(nfp))

        differ=geoFunc.polyToArr(differ_region)

        differ_index=self.getBottomLeft(differ)
        refer_pt_index=geoFunc.checkBottom(adjoin)
        geoFunc.slideToPoint(self.polygons[index],adjoin[refer_pt_index],differ[differ_index])
    
    def getBottomLeft(self,poly):
        bl=[] # bottom left的全部点
        min_y=999999
        # 采用重心最低的原则
        for i,pt in enumerate(poly):
            pt_object={
                    "index":i,
                    "x":pt[0],
                    "y":pt[1]
            }
            if pt[1]<min_y:
                # 如果y更小，那就更新bl
                min_y=pt[1]
                bl=[pt_object]
            elif pt[1]==min_y:
                # 相同则添加这个点
                bl.append(pt_object)
            else:
                pass
        if len(bl)==1:
            return bl[0]["index"]
        else:
            min_x=bl[0]["x"]
            one_pt=bl[0]
            for pt_index in range(1,len(bl)):
                if bl[pt_index]["x"]<min_x:
                    one_pt=bl[pt_index]
                    min_x=one_pt["x"]
            return one_pt["index"]

    def showAll(self):
        '''
        显示最终的排样结果
        '''
        for i in range(0,len(self.polygons)):
            self.gCV.addPolygon(self.polygons[i])
        self.gCV.showPolygon()
    
    def placeFirstPoly(self):
        '''
        放置第一个形状进去，并平移到left-bottom
        '''
        poly=self.polygons[0]
        poly_index=geoFunc.checkBottom(poly)
        left_index,bottom_index,right_index,top_index=geoFunc.checkBound(poly)
        
        move_x=poly[left_index][0]
        move_y=poly[top_index][1]
        geoFunc.slidePoly(poly,0,-move_y)

    def getInnerFitRectangle(self,poly):
        '''
        获得IFR，同时平移到left-bottom
        '''
        poly_index=geoFunc.checkBottom(poly)
        left_index,bottom_index,right_index,top_index=geoFunc.checkBound(poly)
        
        move_x=poly[left_index][0]
        move_y=poly[top_index][1]
        geoFunc.slidePoly(poly,-move_x,-move_y)

        refer_pt=[poly[poly_index][0],poly[poly_index][1]]
        width=self.width-poly[right_index][0]
        height=self.height-poly[bottom_index][1]

        IFP=[refer_pt,[refer_pt[0]+width,refer_pt[1]],[refer_pt[0]+width,refer_pt[1]+height],[refer_pt[0],refer_pt[1]+height]]
        
        return IFP
    
    def getInnerFitRectangleNew(self,poly):
        '''
        获得IFR，不平移
        '''
        poly_index=geoFunc.checkBottom(poly)
        left_index,bottom_index,right_index,top_index=geoFunc.checkBound(poly)
        
        move_x=poly[left_index][0]
        move_y=poly[top_index][1]
        new_poly=geoFunc.getSlide(poly,[-move_x,-move_y])

        refer_pt=[new_poly[poly_index][0],new_poly[poly_index][1]]
        width=self.width-new_poly[right_index][0]
        height=self.height-new_poly[bottom_index][1]

        IFR=[refer_pt,[refer_pt[0]+width,refer_pt[1]],[refer_pt[0]+width,refer_pt[1]+height],[refer_pt[0],refer_pt[1]+height]]
        print("计算出结果:",IFR)

        return IFR
    
    # 形状收缩
    def normData(self,num):
        for poly in self.polygons:
            for ver in poly:
                ver[0]=ver[0]*num
                ver[1]=ver[1]*num

# 计算NFP然后寻找最合适位置
def tryNFP():
    df = pd.read_csv("euro_data/mao.csv")

    poly1=json.loads(df['polygon'][2])
    poly2=json.loads(df['polygon'][3])
    # geoFunc.normData(poly1,50)
    # geoFunc.normData(poly2,50)
    geoFunc.slidePoly(poly1,500,500)

    # nfp=NFP(poly1,poly2,True)
    gCV=graphCV()
    gCV.addPolygon(poly1)
    gCV.addPolygon(poly2)
    gCV.showPolygon()

def getPolygons():
    df = pd.read_csv("euro_data/blaz.csv")
    polygons=[]
    for i in range(0,5):
        polygons.append(json.loads(df['polygon'][i]))
        polygons.append(json.loads(df['polygon'][i]))
        polygons.append(json.loads(df['polygon'][i]))
        polygons.append(json.loads(df['polygon'][i]))
        # polygons.append(json.loads(df['polygon'][i]))
    return polygons

if __name__ == '__main__':
    # tryNFP()
    placeP=PlacePolygons()
    placeP.placePolygons(getPolygons(),True)
    # pass