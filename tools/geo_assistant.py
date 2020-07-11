from shapely.geometry import Polygon,Point,mapping,LineString
from tools.polygon import PltFunc
from math import sqrt
import time

bias = 0.00001

class GeometryAssistant(object):
    '''
    几何相关的算法重新统一
    '''
    @staticmethod
    def getPtNFPPD(pt, convex_status, nfp, pd_bias):
        '''根据最终属性求解PD'''
        min_pd, edges = 999999999, GeometryAssistant.getPolyEdges(nfp)
        last_num = 4 # 最多求往后的3条边
        for k in range(len(edges)):
            # 求解直线边界PD
            nfp_pt, edge = nfp[k], edges[k]
            foot_pt = GeometryAssistant.getFootPoint(pt,edge[0],edge[1]) # 求解垂足
            if GeometryAssistant.bounds(foot_pt[0], edge[0][0], edge[1][0]) == False or GeometryAssistant.bounds(foot_pt[1], edge[0][1], edge[1][1]) == False:
                continue
            pd = sqrt(pow(foot_pt[0]-pt[0],2) + pow(foot_pt[1]-pt[1],2))
            if pd < min_pd:
                min_pd = pd 
            # 求解凹点PD
            if convex_status[k] == 0:
                non_convex_pd = abs(pt[0]-nfp_pt[0]) + abs(pt[1]-nfp_pt[1])
                if non_convex_pd < min_pd:
                    min_pd = non_convex_pd
            # 如果开启了凹点
            if min_pd < 20:
                last_num = last_num - 1
            if last_num == 0:
                break
            # 判断是否为0（一般不会出现该情况）
            if min_pd < pd_bias:
                return 0
        return min_pd

    @staticmethod
    def bounds(val, bound0, bound1):
        if min(bound0, bound1) - bias <= val <= max(bound0, bound1) + bias:
            return True
        else:
            return False

    @staticmethod
    def getLineCoe(line):
        x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
        k = (y2 - y1)/(x2 - x1)
        b = y1 - k*x1
        return k, b

    @staticmethod
    def parallelInter(line1, line2):
        # 判断是否水平，水平用x做参考，其他用y
        k = 1
        if line1[0][1] == line1[1][1] or line2[0][1] == line2[1][1]:
            k = 0
        # 第一个点的包含（不考虑为点）
        if GeometryAssistant.bounds(line1[0][k], line2[0][k], line2[1][k]) == True:
            if GeometryAssistant.bounds(line1[1][k], line2[0][k], line2[1][k]) == True:
                return [line1[0], line1[1]], True # 返回中间的直线
            else:
                if GeometryAssistant.bounds(line2[0][k], line1[0][k], line1[1][k]) == True:
                    return [line1[0], line2[0]], True
                else:
                    return [line1[0], line2[1]], True

        # 没有包含第一个点，判断第二个
        if GeometryAssistant.bounds(line1[1][k], line2[0][k], line2[1][k]) == True:
            if GeometryAssistant.bounds(line2[0][k],line1[0][k], line1[1][k]) == True:
                return [line1[1], line2[0]], True
            else:
                return [line1[1], line2[1]], True

        # Vectical没有包含Line的两个点
        if GeometryAssistant.bounds(line2[0][k], line1[0][k], line1[1][k]) == True:
            return [line2[0], line2[1]], True
        else:
            return [], False

    @staticmethod
    def verticalInter(ver_line, line):
        # 如果另一条直线也垂直
        if line[0][0] == line[1][0]:
            if line[0][0] == ver_line[0][0]:
                return GeometryAssistant.parallelInter(line, ver_line)
            else:
                return [], False
        # 否则求解直线交点
        k, b = GeometryAssistant.getLineCoe(line)
        x = ver_line[0][0]
        y = k * x + b
        if GeometryAssistant.bounds(y, ver_line[0][1], ver_line[1][1]):
            return [[x,y]], True
        else:
            return [], False

    @staticmethod
    def lineInter(line1, line2):
        if min(line1[0][0],line1[1][0]) > max(line2[0][0],line2[1][0]) or max(line1[0][0],line1[1][0]) < min(line2[0][0],line2[1][0]) or min(line1[0][1],line1[1][1]) > max(line2[0][1],line2[1][1]) or max(line1[0][1],line1[1][1]) < min(line2[0][1],line2[1][1]):
            return [], False
        # 为点的情况（例外）
        if line1[0] == line1[1] or line2[0] == line2[1]:
            return [], False
        # 出现直线垂直的情况（没有k）
        if line1[0][0] == line1[1][0]:
            return GeometryAssistant.verticalInter(line1,line2)
        if line2[0][0] == line2[1][0]:
            return GeometryAssistant.verticalInter(line2,line1)
        # 求解y=kx+b
        k1, b1 = GeometryAssistant.getLineCoe(line1)
        k2, b2 = GeometryAssistant.getLineCoe(line2)
        if k1 == k2:
            if b1 == b2:
                return GeometryAssistant.parallelInter(line1, line2)
            else:
                return [], False
        # 求直线交点
        x = (b2 - b1)/(k1 - k2)
        y = k1 * x + b1
        if GeometryAssistant.bounds(x, line1[0][0], line1[1][0]) and GeometryAssistant.bounds(x, line2[0][0], line2[1][0]):
            return [[x,y]], True
        return [], False
    
    @staticmethod
    def interBetweenNFPs(nfp1_edges, nfp2_edges, ifr_bounds):
        '''计算直线交点，仅考虑'''
        inter_points, intersects = [], False
        for edge1 in nfp1_edges:
            for edge2 in nfp2_edges:
                pts, inter_or = GeometryAssistant.lineInter(edge1, edge2)
                if inter_or == False:
                    continue
                intersects = True # 只要有直线交点全部认为是
                for pt in pts:
                    if GeometryAssistant.boundsContain(ifr_bounds, pt) == True:
                        inter_points.append([pt[0],pt[1]])
        return inter_points, intersects

    @staticmethod
    def interNFPIFR(nfp, ifr_bounds, ifr_edges):
        final_points = [] # NFP在IFR内部的点及交点
        contain_last, contain_this = False, False
        for i, pt in enumerate(nfp):
            # 第一个点
            if i == 0:
                if GeometryAssistant.boundsContain(ifr_bounds, pt) == True:
                    final_points.append([pt[0], pt[1]])
                    contain_last = True
                continue
            # 后续的点求解
            if GeometryAssistant.boundsContain(ifr_bounds, pt) == True:
                final_points.append([pt[0], pt[1]])
                contain_this = True
            else:
                contain_this = False
            # 只有两个不全在内侧的时候才需要计算交点
            if contain_last != True and contain_this != True:
                for edge in ifr_edges:
                    inter_pts, inter_or = GeometryAssistant.lineInter([nfp[i-1],nfp[i]], edge)
                    if inter_or == True:
                        for inter_pt in inter_pts:
                            final_points.append([inter_pt[0],inter_pt[1]])
        return final_points
    
    @staticmethod
    def boundsContain(bounds, pt):
        if pt[0] >= bounds[0] and pt[0] <= bounds[2] and pt[1] >= bounds[1] and pt[1] <= bounds[3]:
            return True
        return False
        
    @staticmethod
    def getPolysRight(polys):
        _max=0
        for i in range(0,len(polys)):
            [x,y] = GeometryAssistant.getRightPoint(polys[i])
            if x > _max:
                _max=x
        return _max

    @staticmethod
    def kwtGroupToArray(kwt_group, judge_area):
        '''将几何对象转化为数组，以及是否判断面积大小'''
        array = []
        if kwt_group.geom_type == "Polygon":
            array = GeometryAssistant.kwtItemToArray(kwt_group, judge_area)  # 最终结果只和顶点相关
        else:
            for shapely_item in list(kwt_group):
                array = array + GeometryAssistant.kwtItemToArray(shapely_item,judge_area)
        return array   

    @staticmethod
    def kwtItemToArray(kwt_item, judge_area):
        '''将一个kwt对象转化为数组（比如Polygon）'''
        if judge_area == True and kwt_item.area < bias:
            return []
        res = mapping(kwt_item)
        _arr = []
        # 去除重叠点的情况
        # OutputFunc.outputWarning("可能错误：",res)
        if res["coordinates"][0][0] == res["coordinates"][0][-1]:
            for point in res["coordinates"][0][0:-1]:
                _arr.append([point[0],point[1]])
        else:
            for point in res["coordinates"][0]:
                '''暂时搁置'''
                try:
                    _arr.append([point[0],point[1]])
                except BaseException:
                    pass
        return _arr

    @staticmethod
    def getPolyEdges(poly):
        edges = []
        for index,point in enumerate(poly):
            if index < len(poly)-1:
                edges.append([poly[index],poly[index+1]])
            else:
                if poly[index] != poly[0]:
                    edges.append([poly[index],poly[0]])
        return edges

    @staticmethod
    def getInnerFitRectangle(poly, x, y):
        left_pt, bottom_pt, right_pt, top_pt = GeometryAssistant.getBoundPoint(poly) # 获得全部边界点
        intial_pt = [top_pt[0] - left_pt[0], top_pt[1] - bottom_pt[1]] # 计算IFR初始的位置
        ifr_width = x - right_pt[0] + left_pt[0]  # 获得IFR的宽度
        ifr = [[intial_pt[0], intial_pt[1]], [intial_pt[0] + ifr_width, intial_pt[1]], [intial_pt[0] + ifr_width, y], [intial_pt[0], y]]
        return ifr
    
    @staticmethod
    def getIFRWithBounds(poly,x,y):
        left_pt, bottom_pt, right_pt, top_pt = GeometryAssistant.getBoundPoint(poly) # 获得全部边界点
        intial_pt = [top_pt[0] - left_pt[0], top_pt[1] - bottom_pt[1]] # 计算IFR初始的位置
        ifr_width = x - right_pt[0] + left_pt[0]  # 获得IFR的宽度
        ifr = [[intial_pt[0], intial_pt[1]], [intial_pt[0] + ifr_width, intial_pt[1]], [intial_pt[0] + ifr_width, y], [intial_pt[0], y]]
        return ifr, [intial_pt[0],intial_pt[1],intial_pt[0]+ifr_width,y]
    
    @staticmethod
    def getSlide(poly,x,y):
        '''获得平移后的情况'''
        new_vertex=[]
        for point in poly:
            new_point = [point[0]+x,point[1]+y]
            new_vertex.append(new_point)
        return new_vertex

    @staticmethod
    def normData(poly,num):
        for ver in poly:
            ver[0]=ver[0]*num
            ver[1]=ver[1]*num

    @staticmethod
    def slidePoly(poly,x,y):
        '''将对象平移'''
        for point in poly:
            point[0] = point[0] + x
            point[1] = point[1] + y
    
    @staticmethod
    def slideToPoint(poly,pt):
        '''将对象平移'''
        top_pt = GeometryAssistant.getTopPoint(poly)
        x,y = pt[0] - top_pt[0], pt[1] - top_pt[1]
        for point in poly:
            point[0] = point[0] + x
            point[1] = point[1] + y

    @staticmethod
    def getDirectionalVector(vec):
        _len=math.sqrt(vec[0]*vec[0]+vec[1]*vec[1])
        return [vec[0]/_len,vec[1]/_len]

    @staticmethod
    def deleteOnline(poly):
        '''删除两条直线在一个延长线情况'''
        new_poly=[]
        for i in range(-2,len(poly)-2):
            vec1 = GeometryAssistant.getDirectionalVector([poly[i+1][0]-poly[i][0],poly[i+1][1]-poly[i][1]])
            vec2 = GeometryAssistant.getDirectionalVector([poly[i+2][0]-poly[i+1][0],poly[i+2][1]-poly[i+1][1]])
            if abs(vec1[0]-vec2[0])>bias or abs(vec1[1]-vec2[1])>bias:
                new_poly.append(poly[i+1])
        return new_poly

    @staticmethod
    def getTopPoint(poly):
        top_pt,max_y=[],-999999999
        for pt in poly:
            if pt[1]>max_y:
                max_y=pt[1]
                top_pt=[pt[0],pt[1]]
        return top_pt

    @staticmethod
    def getBottomPoint(poly):
        bottom_pt,min_y=[],999999999
        for pt in poly:
            if pt[1] < min_y:
                min_y = pt[1]
                bottom_pt = [pt[0],pt[1]]
        return bottom_pt

    @staticmethod
    def getRightPoint(poly):
        right_pt,max_x = [], -999999999
        for pt in poly:
            if pt[0] > max_x:
                max_x = pt[0]
                right_pt = [pt[0],pt[1]]
        return right_pt

    @staticmethod
    def getLeftPoint(poly):
        left_pt,min_x=[],999999999
        for pt in poly:
            if pt[0]<min_x:
                min_x=pt[0]
                left_pt=[pt[0],pt[1]]
        return left_pt

    @staticmethod
    def getBottomLeftPoint(poly):
        bottom_left_pt,min_x,min_y=[],999999999,999999999
        for pt in poly:
            if pt[0]<=min_x and pt[1]<=min_y:
                min_x,min_y=pt[0],pt[1]
                bottom_left_pt=[pt[0],pt[1]]
        return bottom_left_pt

    @staticmethod
    def getBoundPoint(poly):
        left_pt,bottom_pt,right_pt,top_pt=[],[],[],[]
        min_x,min_y,max_x,max_y=999999999,999999999,-999999999,-999999999
        for pt in poly:
            if pt[0] < min_x:
                min_x = pt[0]
                left_pt = [pt[0],pt[1]]
            if pt[0] > max_x:
                max_x = pt[0]
                right_pt = [pt[0],pt[1]]
            if pt[1] > max_y:
                max_y = pt[1]
                top_pt = [pt[0],pt[1]]
            if pt[1] < min_y:
                min_y = pt[1]
                bottom_pt = [pt[0],pt[1]]
        return left_pt,bottom_pt,right_pt,top_pt

    @staticmethod
    def getFootPoint(point, line_p1, line_p2):
        """
        @point, line_p1, line_p2 : [x, y, z]
        """
        x0 = point[0]
        y0 = point[1]
    
        x1 = line_p1[0]
        y1 = line_p1[1]
    
        x2 = line_p2[0]
        y2 = line_p2[1]
    
        k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)*1.0
    
        xn = k * (x2 - x1) + x1
        yn = k * (y2 - y1) + y1
    
        return (xn, yn)

class OutputFunc(object):
    '''输出不同颜色字体'''
    @staticmethod
    def outputWarning(prefix,_str):
        '''输出红色字体'''
        _str = prefix + str(time.strftime("%H:%M:%S", time.localtime())) + " " + str(_str)
        print("\033[0;31m",_str,"\033[0m")

    @staticmethod
    def outputAttention(prefix,_str):
        '''输出绿色字体'''
        _str = prefix + str(time.strftime("%H:%M:%S", time.localtime())) + " " + str(_str)
        print("\033[0;32m",_str,"\033[0m")

    @staticmethod
    def outputInfo(prefix,_str):
        '''输出浅黄色字体'''
        _str = prefix + str(time.strftime("%H:%M:%S", time.localtime())) + " " + str(_str)
        print("\033[0;33m",_str,"\033[0m")

class Partition:
    # def __init__(self):
    #     pass
    class PartitionVertex:
        def __init__(self, isActive, p):
            self.isActive=isActive
            self.isConvex=False
            self.isEar=False
            self.p=p
            self.angle=0
            self.previousv=None
            self.nextv=None
        def SetNeighbour(self, previousv, nextv):
            self.previousv=previousv
            self.nextv=nextv

    def IsConvex(self, p1, p2, p3):
        tmp = (p3[1]-p1[1])*(p2[0]-p1[0])-(p3[0]-p1[0])*(p2[1]-p1[1])
        return tmp>0

    def IsReflex(self, p1, p2, p3):
        tmp = (p3[1]-p1[1])*(p2[0]-p1[0])-(p3[0]-p1[0])*(p2[1]-p1[1])
        return tmp<0

    # def IsAcute(self,p1,p2,p3):
    #     if self.IsConvex(p1,p2,p3):
    #         cos=((p3.x-p1.x)*(p2.x-p1.x)+(p3.y-p1.y)*(p2.y-p1.y))/(self.Distance(p2,p1)*self.Distance(p3,p1))
    #         # print cos
    #         if cos>0.5:
    #             return True
    #     return False

    # def InCone(self, p1, p2, p3, p):
    #     convex = self.IsConvex(p1,p2,p3)
    #     if convex:
    #         if not self.IsConvex(p1,p2,p):
    #             return False
    #         if not self.IsConvex(p2,p3,p):
    #             return False
    #         return True
    #     else:
    #         if self.IsConvex(p1,p2,p):
    #             return True
    #         if self.IsConvex(p2,p3,p):
    #             return True
    #         return False

    def Normalize(self, p):
        n = sqrt(p[0]*p[0]+p[1]*p[1])
        if n!=0:
            r = [p[0]/n,p[1]/n]
        else:
            r = Point(0,0)
        return r

    # def Intersects(self,p11,p12,p21,p22):
    #     """check if two lines intersects"""
    #     if p11.x==p21.x and p11.y==p21.y :
    #         return 0
    #     if p11.x==p22.x and p11.y==p22.y :
    #         return 0
    #     if p12.x==p21.x and p12.y==p21.y :
    #         return 0
    #     if p12.x==p22.x and p12.y==p22.y :
    #         return 0
    #     v1ort=Point(p12.y-p11.y,p11.x-p12.x)
    #     v2ort=Point(p22.y-p21.y,p21.x-p22.x)
        
    #     v = p21-p11
    #     dot21 = v.x*v1ort.x + v.y*v1ort.y
    #     v = p22-p11
    #     dot22 = v.x*v1ort.x + v.y*v1ort.y
    #     v = p11-p21
    #     dot11 = v.x*v2ort.x + v.y*v2ort.y
    #     v = p12-p21
    #     dot12 = v.x*v2ort.x + v.y*v2ort.y

    #     if dot11*dot12>0:
    #         return 0
    #     if dot21*dot22>0:
    #         return 0
    #     return 1

    def IsInside(self,p1,p2,p3,p):
        if self.IsConvex(p1,p,p2) or self.IsConvex(p2,p,p3) or self.IsConvex(p3,p,p1):
            return False
        return True

    def UpdateVertex(self,v,vertices):
        v1=v.previousv
        v3=v.nextv
        v.isConvex=self.IsConvex(v1.p, v.p, v3.p)
        vec1=self.Normalize([v1.p[0]-v.p[0],v1.p[1]-v.p[1]])
        vec3=self.Normalize([v3.p[0]-v.p[0],v3.p[1]-v.p[1]])
        v.angle=vec1[0]*vec3[0]+vec1[1]*vec3[1]
        if v.isConvex:
            v.isEar=True
            for vertex in vertices:
                if vertex.p==v.p or vertex.p==v1.p or vertex.p==v3.p:
                    continue
                if self.IsInside(v1.p,v.p,v3.p,vertex.p):
                    v.isEar=False
                    break
        else:
            v.isEar=False

    def getTriangulation(self, poly, triangles):
        '''通过ear clipping进行三角划分 结果存放在triangles中 失败返回False'''
        n=len(poly)
        if n<3 :
            return False
        elif n==3:
            triangles.append(poly)
            return True
        vertices=[]
        for i in range(n):
            vertices.append(self.PartitionVertex(True,poly[i]))
        for i in range(n):
            vertices[i].SetNeighbour(vertices[(i+n-1)%n],vertices[(i+1)%n])
        for vertex in vertices:
            self.UpdateVertex(vertex,vertices)
        for i in range(n-3):
            earfound=False
            ear=None
            for vertex in vertices:
                if not vertex.isActive:
                    continue
                if not vertex.isEar:
                    continue
                if not earfound:
                    earfound=True
                    ear=vertex
                else:
                    if vertex.angle>ear.angle:
                        ear=vertex
            if not earfound:
                return 0
            triangle=[]
            for pt in [ear.previousv.p,ear.p,ear.nextv.p]:
                triangle.append(pt)
            triangles.append(triangle)
            ear.isActive=False
            ear.previousv.nextv=ear.nextv
            ear.nextv.previousv=ear.previousv
            if i==n-4:
                break
            self.UpdateVertex(ear.previousv,vertices)
            self.UpdateVertex(ear.nextv,vertices)
        for vertex in vertices:
            if vertex.isActive:
                triangle=[]
                for pt in [vertex.previousv.p,vertex.p,vertex.nextv.p]:
                    triangle.append(pt)
                triangles.append(triangle)
        return 1

    def getConvexDecomposition(self, poly, parts):
        # triangulate first
        triangles=[]
        if not self.getTriangulation(poly,triangles):
            return 0
        i1=0
        while i1 < len(triangles):
            poly1=triangles[i1]
            i11=i12=i22=i21=-1
            while i11 < len(poly1)-1:
                i11+=1
                d1=poly1[i11]
                i12=(i11+1)%len(poly1)
                d2=poly1[i12]
                isdiagonal=False
                for i2 in range(i1,len(triangles)):
                    if i1==i2:
                        continue
                    poly2=triangles[i2]
                    for i21 in range(len(poly2)):
                        if d2!=poly2[i21]:
                            continue
                        i22=(i21+1)%len(poly2)
                        if d1!=poly2[i22]:
                            continue
                        isdiagonal=True
                        # update adjacent list here
                        break
                    if isdiagonal:
                        break
                if not isdiagonal:
                    continue

                i13=(i11+len(poly1)-1)%len(poly1)
                d3=poly1[i13]
                i14=(i12+1)%len(poly1)
                d4=poly1[i14]
                i23=(i21+len(poly2)-1)%len(poly2)
                d5=poly2[i23]
                i24=(i22+1)%len(poly2)
                d6=poly2[i24]
                if self.IsReflex(d3,d1,d6) or self.IsReflex(d5,d2,d4):
                    continue
                newpoly=[]
                if i12<i11:
                    l=poly1[i12:i11]
                else:
                    l=poly1[i12:]+poly1[:i11]
                for p in l:
                    newpoly.append(p)
                if i22<i21:
                    l=poly2[i22:i21]
                else:
                    l=poly2[i22:]+poly2[:i21]
                for p in l:
                    newpoly.append(p)
                del triangles[i2]
                triangles[i1]=newpoly
                poly1=newpoly
                i1=0
                break
                continue
            i1+=1
        for triangle in triangles:
            parts.append(triangle)
        return 1
