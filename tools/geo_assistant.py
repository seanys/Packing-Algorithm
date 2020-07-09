from shapely.geometry import Polygon,Point,mapping,LineString
from tools.polygon import PltFunc
import math
import time
import copy

bias = 0.00001

class GeometryAssistant(object):
    '''
    几何相关的算法重新统一
    '''
    @staticmethod
    def bounds(val, bound0, bound1):
        if min(bound0, bound1)-bias <= val <= max(bound0, bound1) + bias:
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
        if pt[0] > bounds[0] and pt[0] < bounds[2] and pt[1] > bounds[1] and pt[1] < bounds[3]:
            return True
        return False
        
    @staticmethod
    def getPolysRight(polys):
        _max=0
        for i in range(0,len(polys)):
            [x,y] = GeometryAssistant.getRightPoint(polys[i])
            if x>_max:
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
        if res["coordinates"][0][0] == res["coordinates"][0][-1]:
            for point in res["coordinates"][0][0:-1]:
                _arr.append([point[0],point[1]])
        else:
            for point in res["coordinates"][0]:
                _arr.append([point[0],point[1]])
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
    def getInnerFitRectangle(poly,x,y):
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
        right_pt,max_x=[],-999999999
        for pt in poly:
            if pt[0]>max_x:
                max_x=pt[0]
                right_pt=[pt[0],pt[1]]
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
            if pt[0]<min_x:
                min_x=pt[0]
                left_pt=[pt[0],pt[1]]
            if pt[0]>max_x:
                max_x=pt[0]
                right_pt=[pt[0],pt[1]]
            if pt[1]>max_y:
                max_y=pt[1]
                top_pt=[pt[0],pt[1]]
            if pt[1]<min_y:
                min_y=pt[1]
                bottom_pt=[pt[0],pt[1]]
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