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
