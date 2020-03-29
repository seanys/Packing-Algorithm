'''
Get feasible result based on minimizing overlap and relocating pieces
Several algorithms are achieved in this file
命名规范：类名全部大写、其他函数首字母小写、变量名全部小写
形状情况：计算NFP、BottomLeftFill等均不能影响原始对象
'''
from tools.polygon import GeoFunc,PltFunc,getData,getConvex
from tools.heuristic import TOPOS,BottomLeftFill
from tools.packing import PolyListProcessor,NFPAssistant
import pandas as pd
import json
from shapely.geometry import Polygon,Point,mapping,LineString
from interval import Interval
import copy
import random

precision_error=0.000000001

class Tabu():
    '''
    参考资料：Extended local search algorithm based on nonlinear programming for two-dimensional irregular strip packing problem
    '''
    def __init__(self):
        pass

class GuidedCuckooSearch(object):
    def __init__(self, polygons):
        self.polygons = polygons  # 初始解
        self.n_polys = len(self.polygons)
        self.r_dec = 0.1  # 矩形长度减少百分比
        self.r_inc = 0.1  # 矩形长度增加百分比
        self.W = 1500  # 矩形宽度（固定值）
        self.n_c = 20  # 每代布谷鸟的个数
        self.n_mo = 20  # MinimizeOverlap的最大迭代次数
        self.maxGen = 10  # max generations
        self.penalty = np.ones((self.n_polys, self.n_polys))  # penalty weight
        self.depth = np.zeros((self.n_polys, self.n_polys))  # 渗透深度
        self.percentage = 0.5  # 每次迭代时遗弃的巢的比例
        self.bestF = 999999  # 当前最优解
        print('GCS init:', self.n_polys, 'polygons')

    def guidedCuckooSearch(self, H, N):
        '''
        H: 初始矩形高度
        N: 迭代次数限制
        '''
        self.H = H
        H_best = self.H
        n_cur = 0
        while n_cur <= N:
            original_polygons = list(self.polygons)  # 备份当前解
            it = self.MinimizeOverlap(0, 0, 0)
            if it < self.n_mo:  # 可行解
                H_best = self.H
                self.H = (1-self.r_dec)*self.H
                print('H--: ', self.H)
            else:
                # 不可行解 还原之前的解
                self.polygons = original_polygons
                self.H = (1+self.r_inc)*self.H
                print('H++: ', self.H)
            n_cur = n_cur+1
            self.showAll()
        return H_best

    def CuckooSearch(self, poly_id, ori=''):
        '''
        poly_id: 当前多边形index
        ori: 允许旋转的角度
        '''
        cuckoos = []
        poly = self.polygons[poly_id]
        GL_Algo = BottomLeftFill(self.W,self.polygons)
        R = GL_Algo.getInnerFitRectangleNew(poly)  # 为当前多边形计算inner-fit矩形
        i = 0
        while i < self.n_c:  # 产生初始种群
            c = Cuckoo(R)
            if self.censorCuckoo(c) == False:
                continue
            cuckoos.append(c)
            print(c.getXY())
            i = i+1
        bestCuckoo = cuckoos[0]
        t = 0
        while t < self.maxGen:  # 开始搜索
            c_i = random.choice(cuckoos)
            # 通过Levy飞行产生解
            newCuckooFlag = False
            while newCuckooFlag == False:
                newX, newY = self.getCuckoosLevy(1, bestCuckoo)
                c_i.setXY(newX[0], newY[0])
                if self.censorCuckoo(c_i):
                    newCuckooFlag = True
            self.evaluate(poly_id, c_i, ori)
            c_j = random.choice(cuckoos)
            self.evaluate(poly_id, c_j, ori)
            if c_i.getF() < c_j.getF():
                c_j = c_i
                bestCuckoo = c_j
            # 丢弃一部分最坏的巢并在新位置建立新巢
            cuckoos.sort(key=lambda x: x.getF(), reverse=True)
            newX, newY = self.getCuckoosLevy(
                int(self.percentage*len(cuckoos))+1, bestCuckoo)
            newi = 0
            for i in range(int(len(cuckoos)*self.percentage)):
                print('----- 第', str(t+1), '代 // 第', str(i+1), '只 ----')
                if newi >= len(newX):
                    break
                c_new = Cuckoo(R)
                newCuckooFlag = False
                while newCuckooFlag == False:
                    c_new.setXY(newX[newi], newY[newi])
                    if self.censorCuckoo(c_new) == False:
                        newX, newY = self.getCuckoosLevy(
                            int(self.percentage*len(cuckoos))+1, bestCuckoo)
                        newi = 0
                    else:
                        newCuckooFlag = True
                self.evaluate(poly_id, c_new, ori)
                cuckoos[i] = c_new
                if c_new.getF()==0:
                    break
                newi = newi+1
            cuckoos.sort(key=lambda x: x.getF(), reverse=False)
            bestCuckoo = cuckoos[0]
            bestCuckoo.slidePolytoMe(poly)
            print(bestCuckoo.getF(), bestCuckoo.getXY())
            self.bestF = bestCuckoo.getF()
            for i in range(0, self.n_polys):
                PltFunc.addPolygon(self.polygons[i])
            t = t+1
            PltFunc.saveFig(str(t))
        return bestCuckoo

    def MinimizeOverlap(self, oris, v, o):
        '''
        oris: 允许旋转的角度集合
        v: 多边形位置 实际已通过self.polygons得到
        o: 旋转的角度 后期可考虑把多边形封装成类
        '''
        n_polys = self.n_polys
        it = 0
        fitness = 999999
        while it < self.n_mo:
            Q = np.random.permutation(range(n_polys))
            for i in range(n_polys):
                curPoly = self.polygons[Q[i]]
                # 记录原始位置
                top_index = GeoFunc.checkTop(curPoly)
                top = list(curPoly[top_index])
                F = self.evaluate(Q[i])  # 以后考虑旋转
                print('F of',Q[i],':',F)
                v_i = self.CuckooSearch(Q[i])
                self.evaluate(Q[i], v_i)
                F_new = v_i.getF()
                print('new F of',Q[i],':',F)
                if F_new < F:
                    print('polygon', Q[i], v_i.getXY())
                else:
                    # 平移回原位置
                    GeoFunc.slideToPoint(curPoly, curPoly[top_index], top)
            fitness_new = self.evaluateAll()
            if fitness_new == 0:
                return it  # 可行解
            elif fitness_new < fitness:
                fitness = fitness_new
                it = 0
            self.updatePenalty()
            it = it+1
        return it

    def getCuckoosLevy(self, num, best):
        # Levy flights
        # num: 选取点的个数
        # Source: https://blog.csdn.net/zyqblog/article/details/80905019
        beta = 1.5
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        sigma_v = 1
        x0, y0 = best.getXY()
        x_delta = 0
        y_delta = 0
        resX = []
        resY = []
        for i in range(num*10):
            u = np.random.normal(0, sigma_u, 1)
            v = np.random.normal(0, sigma_v, 1)
            s = u / ((abs(v)) ** (1 / beta))
            x_delta = x_delta+s[0]
            u = np.random.normal(0, sigma_u, 1)
            v = np.random.normal(0, sigma_v, 1)
            s = u / ((abs(v)) ** (1 / beta))
            y_delta = y_delta+s[0]
            resX.append(x_delta)
            resY.append(y_delta)
        # 将所得数据缩放至合适的范围
        x_zoom = self.W/(max(resX)-min(resX))
        y_zoom = self.H/(max(resY)-min(resY))
        Levy_x = []
        Levy_y = []
        for x in resX:
            Levy_x.append(x*x_zoom+x0)
        for y in resY:
            Levy_y.append(y*y_zoom+y0)
        choice = random.sample(range(num*10), num)
        choiceX = []
        choiceY = []
        for i in range(num*10):
            if i in choice:
                choiceX.append(Levy_x[i])
                choiceY.append(Levy_y[i])
        return Levy_x, Levy_y

    def evaluate(self, poly_id, cuckoo=None, ori=None):
        F = 0
        poly = self.polygons[poly_id]
        for p in range(self.n_polys):
            # 将当前多边形的Top平移到cuckoo处
            if cuckoo != None:
                cuckoo.slidePolytoMe(poly)
            if self.polygons[p] == poly:
                continue
            F = F+self.getDepth(self.polygons[p],
                                poly, 0, 0)*self.penalty[p][poly_id]
            if F > self.bestF:
                break
        print('F:', F)
        if cuckoo:
            cuckoo.setF(F)
        else:
            return F

    def evaluateAll(self):
        F = 0
        for i in range(self.n_polys):
            for j in range(i+1, self.n_polys):
                depth = self.getDepth(self.polygons[i], self.polygons[j], 0, 0)
                self.depth[i][j] = depth
                self.depth[j][i] = depth
                F = F+depth*self.penalty[i][j]
        print('all_F:', F)
        return F

    def getDepth(self, poly1, poly2, ori1, ori2):
        '''
        固定poly1 滑动poly2
        计算poly2的checkTop到NFP的距离
        '''
        # 旋转暂未考虑
        return NFP(poly1, poly2).getDepth()

    def showAll(self):
        for i in range(0, self.n_polys):
            PltFunc.addPolygon(self.polygons[i])
        PltFunc.showPlt()

    def updatePenalty(self):
        depth_max = self.depth.max()
        for i in range(self.n_polys):
            for j in range(self.n_polys):
                if i == j:
                    continue
                self.penalty[i][j] = self.penalty[i][j] + \
                    self.depth[i][j]/depth_max

    # 检查布谷鸟是否飞出边界
    def censorCuckoo(self, c):
        if c.getXY()[0] > self.W or c.getXY()[1] > self.H or c.getXY()[0] < 0 or c.getXY()[1] < 0:
            return False
        else:
            return True


class Cuckoo(object):
    def __init__(self, IFR):
        self.F = 999999
        # 在IFR中随机选取位置
        xRank = sorted(IFR, key=lambda x: x[0])
        yRank = sorted(IFR, key=lambda x: x[1])
        self.x = random.random()*(xRank[2][0]-xRank[1][0]+1)+xRank[1][0]
        self.y = random.random()*(yRank[2][1]-yRank[1][1]+1)+yRank[1][1]

    def setF(self, F):
        self.F = F

    def getF(self):
        return self.F

    def getXY(self):
        return [self.x, self.y]

    def setXY(self, x, y):
        self.x = x
        self.y = y

    def slidePolytoMe(self, poly):
        top_index = GeoFunc.checkTop(poly)
        top = poly[top_index]
        GeoFunc.slideToPoint(poly, top, self.getXY())

class GetDP(object):
    '''
    获得 Dominate Point
    参考：https://www.sciencedirect.com/science/article/pii/S0377221713005080
    '''
    def __init__(self,NFP,pp):
        self.NFP=NFP

    # 三个函数通过NFP获得DOminate Point
    def getDP(self,pp):

        NFP_extend=self.NFP+self.NFP
        EP=pp.envelope_polygon+pp.envelope_polygon[:1]
        i=0
        j=0
        begin=False
        points_max=[]
        while i<len(EP)-2:
            line=LineString([EP[i],EP[i+1]])
            points=[]
            while True:
                if begin==False:
                    if NFP_extend[j]==EP[i]:
                        begin=True
                        j=j+1
                    else:
                        j=j+1
                        continue
                if begin==True:
                    if NFP_extend[j]!=EP[i+1]:
                        points.append(Point(NFP_extend[j]))
                        j=j+1
                    else:
                        begin=False
                        break
            if len(points)>0:
                points_max.append(self.getFar(points,line))
            i=i+1

        if len(points_max)>0:
            DP=self.getMax(points_max)
            return DP
        else:
            return False

    def getMax(self,points):
        '''
        获得最大的值，可能出现多个点
        '''
        _max=points[0]
        for i in range(1,len(points)):
            if points[i]["distance"]>_max["distance"]:
                _max=points[i]
        return _max

    def getFar(self,points,line):
        '''
        获得离距离最远的，可能出现多个点
        '''
        _max={
            "cord":[0,0],
            "distance":0
        }
        for point in points:
            pt_dis=point.distance(line)
            if pt_dis>_max["distance"]:
                _max["distance"]=pt_dis
                cord=mapping(point)["coordinates"]
                _max["cord"]=[cord[0],cord[1]]
        return _max

class FNS():
    '''
    参考资料：2004 Fast neighborhood search for two- and three-dimensional nesting problems
    概述：通过贪婪算法，分别在xy轴平移寻找轴上相交面积最小的位置
    待完成：旋转、需要依次采用水平移动/垂直移动/旋转、收缩的高度降低
    '''
    def __init__(self,polys):
        self.polys = polys # 初始的输入形状
        self.cur_polys=polys # 当前形状
        self.poly_list=[] # 包含具体参数，跟随cur_polys变化
        self.width = 1000
        self.height = 999999999
        self.initial()
        self.main()

    def main(self):
        self.shrink()
        self.showResult(current=True)

        self.guidedLocalSearch()

        # while(True):
            # self.shrink()
            # self.guidedLocalSearch()
        
        self.showResult(current=True)
        
    # 获得初始解和判断角度位置  
    def initial(self):
        pp = BottomLeftFill(self.width,self.cur_polys)
        self.height = pp.contain_height
        self.updatePolyList()

    # 收缩宽度，注意cur_polys和poly_list不一样！！！
    def shrink(self):
        self.new_height=self.height*0.95
        for poly in self.cur_polys:
            top_index=GeoFunc.checkTop(poly)
            delta=self.new_height-poly[top_index][1]
            # 如果有重叠就平移
            if delta<0:
                GeoFunc.slidePoly(poly,0,delta)
        self.updatePolyList()
    
    # 展示最终结果
    def showResult(self,**kw):
        if "current" in kw and kw["current"]==True:
            for poly in self.cur_polys:
                PltFunc.addPolygonColor(poly)
            PltFunc.addLine([[0,self.new_height],[self.width,self.new_height]],color="blue")
        if "initial" in kw and kw["initial"]==True:
            for poly in self.polys:
                PltFunc.addPolygon(poly)
            PltFunc.addLine([[0,self.height],[self.width,self.height]],color="blue")
        print(self.polys[0])
        PltFunc.showPlt()
    
    # 获得面积的徒刑
    def overlapCompare(self):
        min_overlap=999999999
        min_t=0
        min_area_t=[]
        for t in self.t_lists:
            overlap=self.getArea(t)
            print("t,overlap:",t,overlap)
            if overlap<min_overlap:
                min_area_t=t
                min_overlap=overlap
        return min_area_t
    
    def getArea(self,t):
        area=0
        for item in self.break_points_list:
            if t>=item[1]:
                area=area+self.getQuadratic(t,item[3][1],item[3][2],item[3][3])
            elif t>=item[0]:
                area=area+self.getQuadratic(t,item[2][1],item[2][2],item[2][3])
            else:
                pass
        return area
    
    # 在水平或者垂直直线上寻找最优位置
    def slideNeighbor(self,poly,_type):
        print("检索类型",_type,"...")
        self.break_points_list=[]
        self.t_lists=[]

        self.getBreakPointList(self.horizontal_positive,self.slide_horizontal_positive,_type,-1)
        self.getBreakPointList(self.horizontal_negative,self.slide_horizontal_negative,_type,-1)
        self.getBreakPointList(self.horizontal_positive,self.slide_horizontal_negative,_type,1)
        self.getBreakPointList(self.horizontal_negative,self.slide_horizontal_positive,_type,1)

        # 计算面积的最合适位置
        self.t_lists=self.chooseFeasible(self.t_lists,_type)
        self.t_lists=self.deleteDuplicated(self.t_lists)
        min_area_t=self.overlapCompare()
        
        print("min_area_t:",min_area_t)
        if abs(min_area_t)<precision_error:
            print("0 未检索到更优位置")
            return False

        # 进行平移位置
        if _type=="vertical":
            GeoFunc.slidePoly(self.cur_polys[self.max_miu_index],0,min_area_t)
        else:
            GeoFunc.slidePoly(self.cur_polys[self.max_miu_index],min_area_t,0)

        # 更新PolyList、正负边、重合情况
        self.updatePolyList()
        self.updateEdgesPN()

        print("1 检索到更优位置")
        self.showResult(current=True)
        return True

    def chooseFeasible(self,_list,_type):
        bounds=Polygon(self.cur_polys[self.max_miu_index]).bounds # min_x min_y max_x max_y
        min_max=[-bounds[0],self.width-bounds[2]]
        if _type=="vertical":
            min_max=[-bounds[1],self.new_height-bounds[3]]
        sorted_list=sorted(_list)
        # 如果超出，那就增加边界的t，以下顺序不能调换
        if sorted_list[-1]>min_max[1]:
            sorted_list.append(min_max[1])
        if sorted_list[0]<min_max[0]:
            sorted_list.append(min_max[0])
        new_list=[]
        for t in sorted_list:
            if t>=min_max[0] and t<=min_max[1]:
                new_list.append(t)
        return new_list

    def deleteDuplicated(self,_list):
        result_list = []
        for item in _list:
            if not item in result_list:
                result_list.append(item)
        return result_list

    # 输入Positive和Negative的边，返回Break Point List
    def getBreakPointList(self,edges,slide_edges,_type,sign):
        for edge in edges:
            for slide_edge in slide_edges:
                res=self.getBreakPoints(edge,slide_edge,_type)
                if res==None:
                    continue
                # 均为Negative或Positive需要为负
                if sign==-1:
                    for ABC in res:
                        for i in range(1,4):
                            ABC[i]=-ABC[i]
                self.t_lists.append(res[0][0])
                self.t_lists.append(res[1][0])
                self.break_points_list.append([res[0][0],res[1][0],res[0],res[1]])

    # 获得水平或垂直平移的情况
    def getBreakPoints(self,edge,slide_edge,_type):
        int_type=0
        if _type=="vertical":
            int_type=1

        # 两条直线四个组合计算
        break_points=[]
        self.getSlideT(slide_edge[0],edge,int_type,1,break_points)
        self.getSlideT(slide_edge[1],edge,int_type,1,break_points)
        self.getSlideT(edge[0],slide_edge,int_type,-1,break_points)
        self.getSlideT(edge[1],slide_edge,int_type,-1,break_points)

        # 必须是有两个交点
        if len(break_points)<2:
            return 
        print(break_points)
        break_points=self.deleteDuplicated(break_points)

        # 开始计算具体参数
        t1=min(break_points[0],break_points[1])
        t2=max(break_points[0],break_points[1])

        sliding_result1=GeoFunc.getSlideLine(slide_edge,t1,0)
        sliding_result2=GeoFunc.getSlideLine(slide_edge,t2,0)
        if _type=="vertical":
            sliding_result1=GeoFunc.getSlideLine(slide_edge,0,t1)
            sliding_result2=GeoFunc.getSlideLine(slide_edge,0,t2)       

        pt1=GeoFunc.intersection(sliding_result1,edge) # 可能为Tuple
        pt2=GeoFunc.intersection(sliding_result2,edge) # 可能为Tuple

        pt3=self.getHoriVerInter(pt1,sliding_result2,int_type)

        ratio=(LineString([pt1,pt2]).length)/(t2-t1) # 两条边的比例
        sin_theta=abs(pt1[1-int_type]-pt2[1-int_type])/(LineString([pt1,pt2]).length) # 直线与水平的角度
        A1=0.5*ratio*sin_theta
        B1=-2*t1*A1
        C1=t1*t1*A1
    
        # 计算A2 B2 C2
        A2=0
        B2=abs(pt1[1-int_type]-pt2[1-int_type]) # 平行四边形的高度
        C2=Polygon([pt1,pt2,pt3]).area-B2*t2 # 三角形面积
        return [[t1,A1,B1,C1],[t2,0,B2,C2]]

    # 获得平移的t值，sign和计算方向相关
    def getSlideT(self,pt,edge,_type,sign,break_points):
        inter=self.getHoriVerInter(pt,edge,_type)
        if len(inter)==0:
            return
        break_points.append((inter[_type]-pt[_type])*sign)

    '''没有考虑不存在的情况/没有考虑直线垂直和水平情况'''
    # 某一点水平或垂直平移后与某直线的交点
    def getHoriVerInter(self,pt,edge,_type):
        upper_pt=edge[1]
        lower_pt=edge[0]
        if edge[0][1-_type]>edge[1][1-_type]:
            upper_pt=edge[0]
            lower_pt=edge[1]
        if pt[1-_type] in Interval(lower_pt[1-_type], upper_pt[1-_type]):
            # 中间的位置比例
            mid=(upper_pt[1-_type]-pt[1-_type])/(upper_pt[1-_type]-lower_pt[1-_type])
            # mid=(upper_pt[_type]-pt[_type])/(upper_pt[_type]-lower_pt[_type])
            # 水平_type=0，计算的也是x即0
            inter=[0,0]
            inter[_type]=upper_pt[_type]-(upper_pt[_type]-lower_pt[_type])*mid
            inter[1-_type]=pt[1-_type]
            return inter
        return []

    # 旋转后的近邻位置
    def rotationNeighbor(self,poly):
        pass
    
    # 获得Positive和Negative的Edges
    def updateEdgesPN(self):
        # 其他形状的边的情况
        self.horizontal_positive=[]
        self.horizontal_negative=[]
        self.vertical_positive=[]
        self.vertical_negative=[]
        for index,item in enumerate(self.poly_list):
            if index!=self.max_miu_index:
                self.appendEdges(self.horizontal_positive,item["horizontal"]["positive"])
                self.appendEdges(self.horizontal_negative,item["horizontal"]["negative"])
                self.appendEdges(self.vertical_positive,item["vertical"]["positive"])
                self.appendEdges(self.vertical_negative,item["vertical"]["negative"])
        # 平移对象的边的情况
        self.slide_horizontal_positive=[]
        self.slide_horizontal_negative=[]
        self.slide_vertical_positive=[]
        self.slide_vertical_negative=[]
        self.appendEdges(self.slide_horizontal_positive,self.poly_list[self.max_miu_index]["horizontal"]["positive"])
        self.appendEdges(self.slide_horizontal_negative,self.poly_list[self.max_miu_index]["horizontal"]["negative"])
        self.appendEdges(self.slide_vertical_positive,self.poly_list[self.max_miu_index]["vertical"]["positive"])
        self.appendEdges(self.slide_vertical_negative,self.poly_list[self.max_miu_index]["vertical"]["negative"])
    
    def appendEdges(self,target,source):
        for edge in source:
            target.append(edge)

    # 寻找最佳位置
    def bestNeighbor(self,poly):
        res=False
        self.updateEdgesPN()
        # 水平移动效果slideNeighbor
        if self.slideNeighbor(poly,"horizontal")==True:
            res=True
        # 垂直移动
        if self.slideNeighbor(poly,"vertical")==True:
            res=True
        # 旋转
        if self.rotationNeighbor(poly)==True:
            res=True

        return res
    
    # 论文 Algorithm1 防止局部最优
    def guidedLocalSearch(self):
        # 初始化的判断参数
        self.phi = [[0]*len(self.cur_polys) for i in range(len(self.cur_polys))] # 惩罚函数
        self.miu_pair=[[0]*len(self.cur_polys) for i in range(len(self.cur_polys))] # 调整后的重叠情况
        self.miu_each=[0 for i in range(len(self.cur_polys))] # 调整后的重叠情况

        # 判断是否有重叠以及寻找最大miu
        self.updateSearchStatus()

        search_times=0

        # 如果有重叠将
        while self.overlap==True and search_times<5:
            # 检索次数限制用于出循环

            print("最大的index为:",self.max_miu_index)
            while self.bestNeighbor(self.cur_polys[self.max_miu_index])==True:
                self.updateSearchStatus()  # 更新并寻找最大Miu
                
            # 更新对应的Phi值并更新Miu
            self.phi[self.max_miu_pair_indx[0]][self.max_miu_pair_indx[0]]+=1
            self.updateSearchStatus()
            print("最大的index更新为:",self.max_miu_index)

            search_times=search_times+1
                
    # 计算所有形状和其他形状的 Overlap 以及是否没有重叠
    def updateSearchStatus(self):
        # 计算重叠情况
        self.overlap_pair=[[0]*len(self.cur_polys) for i in range(len(self.cur_polys))]
        self.overlap_each=[0 for i in range(len(self.cur_polys))]
        for i in range(0,len(self.cur_polys)-1):
            for j in range(i+1,len(self.cur_polys)):
                Pi=Polygon(self.cur_polys[i])
                Pj=Polygon(self.cur_polys[j])
                overlap_area=GeoFunc.computeInterArea(Pi.intersection(Pj))
                if overlap_area>precision_error:
                    self.overlap_pair[i][j]=self.overlap_pair[i][j]+overlap_area
                    self.overlap_pair[j][i]=self.overlap_pair[i][j]
                    self.overlap_each[i]=self.overlap_each[i]+overlap_area
                    self.overlap_each[j]=self.overlap_each[j]+overlap_area
        
        # 更新是否重叠
        self.overlap=False
        for area in self.overlap_each:
            if area>0:
                self.overlap=True

        # 计算对应的Miu
        max_miu_pair=0
        self.max_miu_pair_indx=[0,0]
        for i in range(0,len(self.cur_polys)):
            for j in range(0,len(self.cur_polys)):
                miu=self.overlap_pair[i][j]/(1+self.phi[i][j])
                self.miu_each[i]=self.miu_each[i]+miu
                if miu>max_miu_pair:
                    self.max_miu_pair_indx=[i,j]
    
        # 获得最大的Miu值
        self.max_miu=0
        self.max_miu_index=-1
        for index,miu in enumerate(self.miu_each):
            if miu>self.max_miu:
                self.max_miu=miu
                self.max_miu_index=index

    # 获得当前所有的边的情况
    def updatePolyList(self):
        self.poly_list=[]
        for i,poly in enumerate(self.cur_polys):
            edges=GeoFunc.getPolyEdges(poly)
            poly_item={
                "index":i,
                "pts":poly,
                "edges":edges,
                "horizontal":{
                    "positive":[],
                    "negative":[],
                    "neutral":[]
                },
                "vertical":{
                    "positive":[],
                    "negative":[],
                    "neutral":[]
                },
            }
            for edge in edges:
                netural=self.judgeNeutral(poly,edge) # 分别获得水平和垂直的计算结果
                for i,cur in enumerate(["horizontal","vertical"]):
                    if netural[i]==1:
                        poly_item[cur]["positive"].append([edge[0],edge[1]])
                    elif netural[i]==-1:
                        poly_item[cur]["negative"].append([edge[0],edge[1]])
                    else:
                        poly_item[cur]["neutral"].append([edge[0],edge[1]])
            self.poly_list.append(poly_item)
        # PltFunc.showPlt()
    
    # 判断是否
    def judgeNeutral(self,poly,edge):
        e=0.000001
        P=Polygon(poly)
        mid=[(edge[0][0]+edge[1][0])/2,(edge[0][1]+edge[1][1])/2]
        positive_contain=[P.contains(Point([mid[0]+e,mid[1]])),P.contains(Point([mid[0],mid[1]+e]))] # 水平移动/垂直移动
        neutral=[1,1] # 水平移动/垂直移动
        for i,contain in enumerate(positive_contain):
            if abs(edge[0][1-i]-edge[1][1-i])<precision_error:
                neutral[i]=0
            elif positive_contain[0]==True:
                neutral[i]=1
            else:
                neutral[i]=-1
        return neutral
    
    def getQuadratic(self,x,A,B,C):
        return A*x*x+B*x+C

class ILSQN():
    '''
    参考资料：2009 An iterated local search algorithm based on nonlinear programming for the irregular strip packing problem
    '''
    def __init__(self,poly_list):
        # 初始设置
        self.width=1500

        # 初始化数据，NFP辅助函数
        polys=PolyListProcessor.getPolysVertices(poly_list)
        self.NFPAssistant=NFPAssistant(polys,get_all_nfp=False)

        # 获得最优解
        blf=BottomLeftFill(self.width,polys,NFPAssistant=self.NFPAssistant)
        self.best_height=blf.contain_height
        self.cur_height=blf.contain_height

        # 当前的poly_list均为已经排样的情况
        self.best_poly_list=copy.deepcopy(poly_list)
        self.cur_poly_list=copy.deepcopy(poly_list)

        self.run()
    
    def run(self):
        for i in range(1):
            if self.minimizeOverlap()==True:
                pass
            else:
                pass
        
    def minimizeOverlap(self):
        k=0
        while k<5:
            initial_solution,height=self.swapTwoPolygons()
            lopt_solution=self.separate(initial_solution)
            pass
    
    def findBestPosition(self):
        pass

    def swapTwoPolygons(self):
        i,j=random.randint(0,len(self.cur_poly_list)-1),random.randint(0,len(self.cur_poly_list)-1)
        pass

    def separate(self):
        pass


class Test():
    
    def testDepth(self):
        polys = getData()
        gcs = GuidedCuckooSearch(polys)
        poly1 = [[500.0, 500.0], [602.75, 516.25], [700.0, 500.0], [797.25, 516.25], [
            900.0, 500.0], [875.0, 592.0], [700.0, 571.5], [525.0, 592.0]]
        poly2 = [[0, 0], [102.75, 16.25], [200.0, 0.0], [297.25, 16.25],
                 [400.0, 0.0], [375.0, 92.0], [200.0, 71.5], [25.0, 92.0]]
        poly3 = [[400.0, 400.0], [602.75, 516.25], [700.0, 500.0], [797.25, 516.25], [
            900.0, 500.0], [875.0, 592.0], [700.0, 571.5], [525.0, 592.0]]
        return gcs.getDepth(poly1, poly3, 0, 0)

    def testGCS(self):
        # polygons=[]
        # polygons.append(self.getTestPolys()[0])
        # polygons.append(self.getTestPolys()[1])
        polygons=getData()
        num = 1  # 形状收缩
        for poly in polygons:
            for ver in poly:
                ver[0] = ver[0]*num
                ver[1] = ver[1]*num
        gcs = GuidedCuckooSearch(polygons)
        GeoFunc.slidePoly(polygons[0], 500, 500)
        gcs.showAll()
        gcs.guidedCuckooSearch(1500, 10)
        gcs.showAll()

    def testLevy(self):
        gcs = GuidedCuckooSearch(None)
        c = Cuckoo([[495.75, 565.0], [745.75, 565.0], [
                   745.75, 2000.0], [495.75, 2000.0]])
        # c.setXY(594.9059139903344, 583.4635636682448)
        c.setXY(500, 500)
        xy = gcs.getCuckoosLevy(10, c)
        plt.plot(xy[0], xy[1])
        plt.show()


if __name__ == "__main__":
    polys=getConvex(num=5)
    nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    Compaction(BottomLeftFill(1500,polys,vertical=True,NFPAssistant=nfp_ass).polygons)
    # ILSQN(poly_list)
