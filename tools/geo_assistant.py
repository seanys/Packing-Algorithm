from shapely.geometry import Polygon,Point,mapping,LineString
from tools.polygon import PltFunc
from math import sqrt,acos
import time
bias = 0.0000001

class GeometryAssistant(object):
    '''
    几何相关的算法重新统一
    '''
    @staticmethod
    def getAdjustPts(original_points, first_pt, to_real):
        '''部分情况需要根据相对位置调整范围'''
        new_points = []
        for pt in original_points:
            if to_real == True:
                new_points.append([pt[0]+first_pt[0],pt[1]+first_pt[1]])
            else:
                new_points.append([pt[0]-first_pt[0],pt[1]-first_pt[1]])
        return new_points

    @staticmethod
    def judgeContain(pt,parts):
        '''判断点是否包含在NFP凸分解后的凸多边形列表中 输入相对位置点'''
        def cross(p0,p1,p2):
            '''计算叉乘'''
            return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

        for part in parts: # 对凸多边形逐个判断
            n=len(part)
            if cross(part[0],pt,part[1])>0 or cross(part[0],pt,part[n-1])<0:
                continue
            i=1
            j=n-1
            line=-1
            while i<=j:
                mid=int((i+j)/2)
                if cross(part[0],pt,part[mid])>0:
                    line=mid
                    j=mid-1
                else:
                    i=mid+1
            if cross(part[line-1],pt,part[line])<=0:
                return True
        return False

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
    def getPointsContained(inter_points, ifr_bounds):
        new_points = []
        for pt in inter_points:
            if GeometryAssistant.boundsContain(ifr_bounds, pt):
                new_points.append(pt)
        return new_points
    
    @staticmethod
    def interBetweenNFPs(nfp1_edges, nfp2_edges):
        '''计算直线交点，仅考虑'''
        inter_points, intersects = [], False
        for edge1 in nfp1_edges:
            for edge2 in nfp2_edges:
                pts, inter_or = GeometryAssistant.lineInter(edge1, edge2)
                if inter_or == False:
                    continue
                intersects = True # 只要有直线交点全部认为是
                for pt in pts:
                    if [pt[0],pt[1]] not in inter_points:
                        inter_points.append([pt[0],pt[1]])
        return inter_points, intersects

    @staticmethod
    def interNFPIFR(nfp, ifr_bounds, ifr_edges, ifr):
        '''求解NFP与IFR的相交区域'''
        total_points, border_pts, inside_indexs = [], [], [] # NFP在IFR内部的点及交，计算参考和边界可行范围
        contain_last, contain_this = False, False
        temp_nfp = nfp + [nfp[0]]
        for i, pt in enumerate(temp_nfp):
            # 第一个点
            if i == 0:
                if GeometryAssistant.boundsContain(ifr_bounds, pt) == True:
                    inside_indexs.append(i)
                    total_points.append([pt[0], pt[1]])
                    contain_last = True
                continue
            # 后续的点求解
            if GeometryAssistant.boundsContain(ifr_bounds, pt) == True and i != len(temp_nfp) - 1:
                inside_indexs.append(i)
                total_points.append([pt[0], pt[1]])
                contain_this = True
            else:
                contain_this = False
            # 只有两个不全在内侧的时候才需要计算交点
            if contain_last == False or contain_this == False:
                for k,edge in enumerate(ifr_edges):
                    inter_pts, inter_or = GeometryAssistant.lineInter([temp_nfp[i-1],temp_nfp[i]], edge)
                    if inter_or == True:
                        for new_pt in inter_pts:
                            if new_pt not in total_points:
                                total_points.append(new_pt) # 将交点加入可行点
                                border_pts.append(new_pt) # 将交点加入可行点
            contain_last = contain_this
        return total_points, inside_indexs, border_pts
    
    @staticmethod
    def addRelativeRecord(record_target, target_key, inside_indexs, border_pts, first_pt):
        adjust_border_pts = GeometryAssistant.getAdjustPts(border_pts, first_pt, False)
        record_target[target_key] = {}
        record_target[target_key]["adjust_border_pts"] = adjust_border_pts
        record_target[target_key]["inside_indexs"] = inside_indexs

    @staticmethod
    def addAbsoluteRecord(record_target, target_key, inside_indexs, border_pts):
        record_target[target_key] = {}
        record_target[target_key]["border_pts"] = border_pts
        record_target[target_key]["inside_indexs"] = inside_indexs

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
        for index in range(len(poly)):
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
        _len=sqrt(vec[0]*vec[0]+vec[1]*vec[1])
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

    @staticmethod
    def judgePositive(pt1, pt2, k):
        if k%2 == 0:
            if pt1[1] > pt2[1]:
                return 1
            elif pt1[1] < pt2[1]:
                return -1
            return 0
        if k%2 == 1:
            if pt2[0] > pt1[0]:
                return 1
            elif pt2[0] < pt1[0]:
                return -1
            return 0
        return 0

    @staticmethod
    def judgeLeft(pt1, pt2):
        x,y = 0,0
        if pt2[1] - pt1[1] > 0:
            x = 1
        elif pt2[1] - pt1[1] < 0:
            x = -1
        if pt2[0] - pt1[0] > 0:
            y = 1
        elif pt2[0] - pt1[0] < 0:
            y = -1
        return x,y

    @staticmethod
    def getAdjustRange(original_border_range, first_pt, to_real):
        '''部分情况需要根据相对位置调整范围'''
        new_border_range = []
        for i in range(4):
            border_range = []
            for item in original_border_range[i]:
                if to_real == True:
                    border_range.append([item[0]+first_pt[i%2],item[1]+first_pt[i%2]])
                else:
                    border_range.append([item[0]-first_pt[i%2],item[1]-first_pt[i%2]])
            new_border_range.append(border_range)
        return new_border_range

    @staticmethod
    def getFeasiblePt(ifr_bound, infeasible_border_range):
        '''求解可行域的中的可行点，从左下角逆时针'''
        potential_points = []
        for k, every_border_range in enumerate(infeasible_border_range):
            all_position = list(set([p for bound in every_border_range for p in bound] + [ifr_bound[k%2],ifr_bound[k%2+2]]))
            for position in all_position:
                feasible = True
                for test_range in every_border_range:
                    if test_range[0] < position < test_range[1] or position > ifr_bound[k%2] or position < ifr_bound[k%2+2]:
                        feasible = False
                        break
                if feasible == True:
                    if k%2 == 0:
                        potential_points.append([position, ifr_bound[k+1]])
                    else:
                        potential_points.append([ifr_bound[3-k], position])
        return potential_points


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

def lineInt(l1, l2, precision=0):
    """Compute the intersection between two lines.
    Keyword arguments:
    l1 -- first line
    l2 -- second line
    precision -- precision to check if lines are parallel (default 0)
    Returns:
    The intersection point
    """
    i = [0, 0] # point
    a1 = l1[1][1] - l1[0][1]
    b1 = l1[0][0] - l1[1][0]
    c1 = a1 * l1[0][0] + b1 * l1[0][1]
    a2 = l2[1][1] - l2[0][1]
    b2 = l2[0][0] - l2[1][0]
    c2 = a2 * l2[0][0] + b2 * l2[0][1]
    det = a1 * b2 - a2 * b1
    if not scalar_eq(det, 0, precision): # lines are not parallel
        i[0] = (b2 * c1 - b1 * c2) / det
        i[1] = (a1 * c2 - a2 * c1) / det
    return i

def lineSegmentsIntersect(p1, p2, q1, q2):
    """Checks if two line segments intersect.
    Keyword arguments:
    p1 -- The start vertex of the first line segment.
    p2 -- The end vertex of the first line segment.
    q1 -- The start vertex of the second line segment.
    q2 -- The end vertex of the second line segment.
    Returns:
    True if the two line segments intersect
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    da = q2[0] - q1[0]
    db = q2[1] - q1[1]

    # segments are parallel
    if (da*dy - db*dx) == 0:
        return False

    s = (dx * (q1[1] - p1[1]) + dy * (p1[0] - q1[0])) / (da * dy - db * dx)
    t = (da * (p1[1] - q1[1]) + db * (q1[0] - p1[0])) / (db * dx - da * dy)

    return s >= 0 and s <= 1 and t >= 0 and t <= 1

def triangleArea(a, b, c):
    """Calculates the area of a triangle spanned by three points.
    Note that the area will be negative if the points are not given in counter-clockwise order.
    Keyword arguments:
    a -- First point
    b -- Second point
    c -- Third point
    Returns:
    Area of triangle
    """
    return ((b[0] - a[0])*(c[1] - a[1]))-((c[0] - a[0])*(b[1] - a[1]))

def isLeft(a, b, c):
    return triangleArea(a, b, c) > 0

def isLeftOn(a, b, c):
    return triangleArea(a, b, c) >= 0

def isRight(a, b, c):
    return triangleArea(a, b, c) < 0

def isRightOn(a, b, c):
    return triangleArea(a, b, c) <= 0

def collinear(a, b, c, thresholdAngle=0):
    """Checks if three points are collinear.
    Keyword arguments:
    a -- First point
    b -- Second point
    c -- Third point
    thresholdAngle -- threshold to consider if points are collinear, in radians (default 0)
    Returns:
    True if points are collinear
    """
    if thresholdAngle == 0:
        return triangleArea(a, b, c) == 0
    else:
        ab = [None] * 2
        bc = [None] * 2

        ab[0] = b[0]-a[0]
        ab[1] = b[1]-a[1]
        bc[0] = c[0]-b[0]
        bc[1] = c[1]-b[1]

        dot = ab[0]*bc[0] + ab[1]*bc[1]
        magA = sqrt(ab[0]*ab[0] + ab[1]*ab[1])
        magB = sqrt(bc[0]*bc[0] + bc[1]*bc[1])
        angle = acos(dot/(magA*magB))
        return angle < thresholdAngle

def sqdist(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return dx * dx + dy * dy

def polygonAt(polygon, i):
    """Gets a vertex at position i on the polygon.
    It does not matter if i is out of bounds.
    Keyword arguments:
    polygon -- The polygon
    i -- Position desired on the polygon
    Returns:
    Vertex at position i
    """
    s = len(polygon)
    return polygon[i % s]

def polygonClear(polygon):
    """Clears the polygon data
    Keyword arguments:
    polygon -- The polygon
    """
    del polygon[:]

def polygonAppend(polygon, poly, start, end):
    """Grabs points at indicies `start` to `end` from `poly`
    and appends them to `polygon`
    Keyword arguments:
    polygon -- The destination polygon
    poly -- The source polygon
    start -- Starting source index
    end -- Ending source index (not included in the slice)
    """
    for i in range(start, end):
        polygon.append(poly[i])

def polygonMakeCCW(polygon):
    """Makes sure that the polygon vertices are ordered counter-clockwise.
    Keyword arguments:
    polygon -- The polygon
    """
    br = 0
    v = polygon

    # find bottom right point
    for i in range(1, len(polygon)):
        if v[i][1] < v[br][1] or (v[i][1] == v[br][1] and v[i][0] > v[br][0]):
            br = i

    # reverse poly if clockwise
    if not isLeft(polygonAt(polygon, br - 1), polygonAt(polygon, br), polygonAt(polygon, br + 1)):
        polygonReverse(polygon)

def polygonReverse(polygon):
    """Reverses the vertices in the polygon.
    Keyword arguments:
    polygon -- The polygon
    """
    polygon.reverse()

def polygonIsReflex(polygon, i):
    """Checks if a point in the polygon is a reflex point.
    Keyword arguments:
    polygon -- The polygon
    i -- index of point to check
    Returns:
    True is point is a reflex point
    """
    return isRight(polygonAt(polygon, i - 1), polygonAt(polygon, i), polygonAt(polygon, i + 1))

def polygonCanSee(polygon, a, b):
    """Checks if two vertices in the polygon can see each other.
    Keyword arguments:
    polygon -- The polygon
    a -- Vertex 1
    b -- Vertex 2
    Returns:
    True if vertices can see each other
    """

    l1 = [None]*2
    l2 = [None]*2

    if isLeftOn(polygonAt(polygon, a + 1), polygonAt(polygon, a), polygonAt(polygon, b)) and isRightOn(polygonAt(polygon, a - 1), polygonAt(polygon, a), polygonAt(polygon, b)):
        return False

    dist = sqdist(polygonAt(polygon, a), polygonAt(polygon, b))
    for i in range(0, len(polygon)): # for each edge
        if (i + 1) % len(polygon) == a or i == a: # ignore incident edges
            continue

        if isLeftOn(polygonAt(polygon, a), polygonAt(polygon, b), polygonAt(polygon, i + 1)) and isRightOn(polygonAt(polygon, a), polygonAt(polygon, b), polygonAt(polygon, i)): # if diag intersects an edge
            l1[0] = polygonAt(polygon, a)
            l1[1] = polygonAt(polygon, b)
            l2[0] = polygonAt(polygon, i)
            l2[1] = polygonAt(polygon, i + 1)
            p = lineInt(l1, l2)
            if sqdist(polygonAt(polygon, a), p) < dist: # if edge is blocking visibility to b
                return False

    return True

def polygonCopy(polygon, i, j, targetPoly=None):
    """Copies the polygon from vertex i to vertex j to targetPoly.
    Keyword arguments:
    polygon -- The source polygon
    i -- start vertex
    j -- end vertex (inclusive)
    targetPoly -- Optional target polygon
    Returns:
    The resulting copy.
    """
    p = targetPoly or []
    polygonClear(p)
    if i < j:
        # Insert all vertices from i to j
        for k in range(i, j+1):
            p.append(polygon[k])

    else:
        # Insert vertices 0 to j
        for k in range(0, j+1):
            p.append(polygon[k])

        # Insert vertices i to end
        for k in range(i, len(polygon)):
            p.append(polygon[k])

    return p

def polygonGetCutEdges(polygon):
    """Decomposes the polygon into convex pieces.
    Note that this algorithm has complexity O(N^4) and will be very slow for polygons with many vertices.
    Keyword arguments:
    polygon -- The polygon
    Returns:
    A list of edges [[p1,p2],[p2,p3],...] that cut the polygon.
    """
    mins = []
    tmp1 = []
    tmp2 = []
    tmpPoly = []
    nDiags = float('inf')

    for i in range(0, len(polygon)):
        if polygonIsReflex(polygon, i):
            for j in range(0, len(polygon)):
                if polygonCanSee(polygon, i, j):
                    tmp1 = polygonGetCutEdges(polygonCopy(polygon, i, j, tmpPoly))
                    tmp2 = polygonGetCutEdges(polygonCopy(polygon, j, i, tmpPoly))

                    for k in range(0, len(tmp2)):
                        tmp1.append(tmp2[k])

                    if len(tmp1) < nDiags:
                        mins = tmp1
                        nDiags = len(tmp1)
                        mins.append([polygonAt(polygon, i), polygonAt(polygon, j)])

    return mins

def polygonDecomp(polygon):
    """Decomposes the polygon into one or more convex sub-polygons.
    Keyword arguments:
    polygon -- The polygon
    Returns:
    An array or polygon objects.
    """
    edges = polygonGetCutEdges(polygon)
    if len(edges) > 0:
        return polygonSlice(polygon, edges)
    else:
        return [polygon]

def polygonSlice(polygon, cutEdges):
    """Slices the polygon given one or more cut edges. If given one, this function will return two polygons (false on failure). If many, an array of polygons.
    Keyword arguments:
    polygon -- The polygon
    cutEdges -- A list of edges to cut on, as returned by getCutEdges()
    Returns:
    An array of polygon objects.
    """
    if len(cutEdges) == 0:
        return [polygon]

    if isinstance(cutEdges, list) and len(cutEdges) != 0 and isinstance(cutEdges[0], list) and len(cutEdges[0]) == 2 and isinstance(cutEdges[0][0], list):

        polys = [polygon]

        for i in range(0, len(cutEdges)):
            cutEdge = cutEdges[i]
            # Cut all polys
            for j in range(0, len(polys)):
                poly = polys[j]
                result = polygonSlice(poly, cutEdge)
                if result:
                    # Found poly! Cut and quit
                    del polys[j:j+1]
                    polys.extend((result[0], result[1]))
                    break

        return polys
    else:

        # Was given one edge
        cutEdge = cutEdges
        i = polygon.index(cutEdge[0])
        j = polygon.index(cutEdge[1])

        if i != -1 and j != -1:
            return [polygonCopy(polygon, i, j),
                    polygonCopy(polygon, j, i)]
        else:
            return False

def polygonIsSimple(polygon):
    """Checks that the line segments of this polygon do not intersect each other.
    Keyword arguments:
    polygon -- The polygon
    Returns:
    True is polygon is simple (not self-intersecting)
    Todo:
    Should it check all segments with all others?
    """
    path = polygon
    # Check
    for i in range(0,len(path)-1):
        for j in range(0, i-1):
            if lineSegmentsIntersect(path[i], path[i+1], path[j], path[j+1]):
                return False

    # Check the segment between the last and the first point to all others
    for i in range(1,len(path)-2):
        if lineSegmentsIntersect(path[0], path[len(path)-1], path[i], path[i+1]):
            return False

    return True

def getIntersectionPoint(p1, p2, q1, q2, delta=0):
    """Gets the intersection point 
    Keyword arguments:
    p1 -- The start vertex of the first line segment.
    p2 -- The end vertex of the first line segment.
    q1 -- The start vertex of the second line segment.
    q2 -- The end vertex of the second line segment.
    delta -- Optional precision to check if lines are parallel (default 0)
    Returns:
    The intersection point.
    """
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = (a1 * p1[0]) + (b1 * p1[1])
    a2 = q2[1] - q1[1]
    b2 = q1[0] - q2[0]
    c2 = (a2 * q1[0]) + (b2 * q1[1])
    det = (a1 * b2) - (a2 * b1)

    if not scalar_eq(det, 0, delta):
        return [((b2 * c1) - (b1 * c2)) / det, ((a1 * c2) - (a2 * c1)) / det]
    else:
        return [0, 0]

def polygonQuickDecomp(polygon, result=None, reflexVertices=None, steinerPoints=None, delta=25, maxlevel=200, level=0):
    """Quickly decompose the Polygon into convex sub-polygons.
    Keyword arguments:
    polygon -- The polygon to decompose
    result -- Stores result of decomposed polygon, passed recursively
    reflexVertices -- 
    steinerPoints --
    delta -- Currently unused
    maxlevel -- The maximum allowed level of recursion
    level -- The current level of recursion
    Returns:
    List of decomposed convex polygons
    """
    if result is None:
        result = []
    reflexVertices = reflexVertices or []
    steinerPoints = steinerPoints or []

    upperInt = [0, 0]
    lowerInt = [0, 0]
    p = [0, 0]         # Points
    upperDist = 0
    lowerDist = 0
    d = 0
    closestDist = 0 # scalars
    upperIndex = 0
    lowerIndex = 0
    closestIndex = 0 # integers
    lowerPoly = []
    upperPoly = [] # polygons
    poly = polygon
    v = polygon

    if len(v) < 3:
        return result

    level += 1
    if level > maxlevel:
        print("quickDecomp: max level ("+str(maxlevel)+") reached.")
        return result

    for i in range(0, len(polygon)):
        if polygonIsReflex(poly, i):
            reflexVertices.append(poly[i])
            upperDist = float('inf')
            lowerDist = float('inf')

            for j in range(0, len(polygon)):
                if isLeft(polygonAt(poly, i - 1), polygonAt(poly, i), polygonAt(poly, j)) and isRightOn(polygonAt(poly, i - 1), polygonAt(poly, i), polygonAt(poly, j - 1)): # if line intersects with an edge
                    p = getIntersectionPoint(polygonAt(poly, i - 1), polygonAt(poly, i), polygonAt(poly, j), polygonAt(poly, j - 1)) # find the point of intersection
                    if isRight(polygonAt(poly, i + 1), polygonAt(poly, i), p): # make sure it's inside the poly
                        d = sqdist(poly[i], p)
                        if d < lowerDist: # keep only the closest intersection
                            lowerDist = d
                            lowerInt = p
                            lowerIndex = j

                if isLeft(polygonAt(poly, i + 1), polygonAt(poly, i), polygonAt(poly, j + 1)) and isRightOn(polygonAt(poly, i + 1), polygonAt(poly, i), polygonAt(poly, j)):
                    p = getIntersectionPoint(polygonAt(poly, i + 1), polygonAt(poly, i), polygonAt(poly, j), polygonAt(poly, j + 1))
                    if isLeft(polygonAt(poly, i - 1), polygonAt(poly, i), p):
                        d = sqdist(poly[i], p)
                        if d < upperDist:
                            upperDist = d
                            upperInt = p
                            upperIndex = j

            # if there are no vertices to connect to, choose a point in the middle
            if lowerIndex == (upperIndex + 1) % len(polygon):
                #print("Case 1: Vertex("+str(i)+"), lowerIndex("+str(lowerIndex)+"), upperIndex("+str(upperIndex)+"), poly.size("+str(len(polygon))+")")
                p[0] = (lowerInt[0] + upperInt[0]) / 2
                p[1] = (lowerInt[1] + upperInt[1]) / 2
                steinerPoints.append(p)

                if i < upperIndex:
                    #lowerPoly.insert(lowerPoly.end(), poly.begin() + i, poly.begin() + upperIndex + 1)
                    polygonAppend(lowerPoly, poly, i, upperIndex+1)
                    lowerPoly.append(p)
                    upperPoly.append(p)
                    if lowerIndex != 0:
                        #upperPoly.insert(upperPoly.end(), poly.begin() + lowerIndex, poly.end())
                        polygonAppend(upperPoly, poly, lowerIndex, len(poly))

                    #upperPoly.insert(upperPoly.end(), poly.begin(), poly.begin() + i + 1)
                    polygonAppend(upperPoly, poly, 0, i+1)
                else:
                    if i != 0:
                        #lowerPoly.insert(lowerPoly.end(), poly.begin() + i, poly.end())
                        polygonAppend(lowerPoly, poly, i, len(poly))

                    #lowerPoly.insert(lowerPoly.end(), poly.begin(), poly.begin() + upperIndex + 1)
                    polygonAppend(lowerPoly, poly, 0, upperIndex+1)
                    lowerPoly.append(p)
                    upperPoly.append(p)
                    #upperPoly.insert(upperPoly.end(), poly.begin() + lowerIndex, poly.begin() + i + 1)
                    polygonAppend(upperPoly, poly, lowerIndex, i+1)

            else:
                # connect to the closest point within the triangle
                #print("Case 2: Vertex("+str(i)+"), closestIndex("+str(closestIndex)+"), poly.size("+str(len(polygon))+")\n")

                if lowerIndex > upperIndex:
                    upperIndex += len(polygon)

                closestDist = float('inf')

                if upperIndex < lowerIndex:
                    return result

                for j in range(lowerIndex, upperIndex+1):
                    if isLeftOn(polygonAt(poly, i - 1), polygonAt(poly, i), polygonAt(poly, j)) and isRightOn(polygonAt(poly, i + 1), polygonAt(poly, i), polygonAt(poly, j)):
                        d = sqdist(polygonAt(poly, i), polygonAt(poly, j))
                        if d < closestDist:
                            closestDist = d
                            closestIndex = j % len(polygon)

                if i < closestIndex:
                    polygonAppend(lowerPoly, poly, i, closestIndex+1)
                    if closestIndex != 0:
                        polygonAppend(upperPoly, poly, closestIndex, len(v))

                    polygonAppend(upperPoly, poly, 0, i+1)
                else:
                    if i != 0:
                        polygonAppend(lowerPoly, poly, i, len(v))

                    polygonAppend(lowerPoly, poly, 0, closestIndex+1)
                    polygonAppend(upperPoly, poly, closestIndex, i+1)

            # solve smallest poly first
            if len(lowerPoly) < len(upperPoly):
                polygonQuickDecomp(lowerPoly, result, reflexVertices, steinerPoints, delta, maxlevel, level)
                polygonQuickDecomp(upperPoly, result, reflexVertices, steinerPoints, delta, maxlevel, level)
            else:
                polygonQuickDecomp(upperPoly, result, reflexVertices, steinerPoints, delta, maxlevel, level)
                polygonQuickDecomp(lowerPoly, result, reflexVertices, steinerPoints, delta, maxlevel, level)

            return result

    result.append(polygon)

    return result

def polygonRemoveCollinearPoints(polygon, precision=0):
    """Remove collinear points in the polygon.
    Keyword arguments:
    polygon -- The polygon
    precision -- The threshold angle to use when determining whether two edges are collinear. (default is 0)
    Returns:
    The number of points removed
    """
    num = 0
    i = len(polygon) - 1
    while len(polygon) > 3 and i >= 0:
    #(var i=polygon.length-1; polygon.length>3 && i>=0; --i){
        if collinear(polygonAt(polygon, i - 1), polygonAt(polygon, i), polygonAt(polygon, i+1), precision):
            # Remove the middle point
            del polygon[i % len(polygon):(i % len(polygon))+1]
            num += 1
        i -= 1
    return num

def scalar_eq(a, b, precision=0):
    """Check if two scalars are equal.
    Keyword arguments:
    a -- first scalar
    b -- second scalar
    precision -- precision to check equality
    Returns:
    True if scalars are equal
    """
    return abs(a - b) <= precision

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

    def Normalize(self, p):
        n = sqrt(p[0]*p[0]+p[1]*p[1])
        if n!=0:
            r = [p[0]/n,p[1]/n]
        else:
            r = Point(0,0)
        return r

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
            existed=False
            for i in range(len(parts)):
                part=parts[i]
                if len(part)==len(triangle) and Polygon(part).area-Polygon(triangle).area<0.0001:
                    existed=True
                    break
            if not existed:
                parts.append(triangle)
        return 1
