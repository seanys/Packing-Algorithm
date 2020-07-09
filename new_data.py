from tools.polygon import PltFunc,GeoFunc,NFP,getData
from sequence import BottomLeftFill
from tools.geo_assistant import GeometryAssistant
from tools.packing import NFPAssistant
from tools.lp_assistant import LPAssistant
from shapely.geometry import Polygon,mapping,Point
from shapely import affinity
import pandas as pd # 读csv
import csv # 写csv
import json
import itertools
import copy
import math

targets = [{
        "index" : 0,
        "name" : "blaz",
        "scale" : 10,
        "allowed_rotation": 2,
        "width": 150
    },{
        "index" : 1,
        "name" : "shapes2_clus",
        "scale" : 1,
        "allowed_rotation": 2,
        "width": 750
    },{
        "index" : 2,
        "name" : "shapes0",
        "scale" : 20,
        "allowed_rotation": 1,
        "width": 800
    },{
        "index" : 3,
        "name" : "marques",
        "scale" : 10,
        "allowed_rotation": 2,
        "width": 1040
    },{
        "index" : 4,
        "name" : "mao",
        "scale" : 1,
        "allowed_rotation": 4,
        "width": 2550
    },{
        "index" : 5,
        "name" : "shirts",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 800
    },{
        "index" : 6,
        "name" : "albano",
        "scale" : 0.2,
        "allowed_rotation": 2,
        "width": 980
    },{
        "index" : 7,
        "name" : "shapes1",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 800
    },{
        "index" : 8,
        "name" : "dagli_clus",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 1200
    },{
        "index" : 9,
        "name" : "jakobs1_clus",
        "scale" : 20,
        "allowed_rotation": 4,
        "width": 800
    },{
        "index" : 10,
        "name" : "trousers",
        "scale" : 10,
        "allowed_rotation": 2,
        "width": 790
    },{
        "index" : 11,
        "name" : "jakobs2_clus",
        "scale" : 10,
        "allowed_rotation": 4,
        "width": 700
    },{
        "index" : 12,
        "name" : "swim_clus",
        "scale" : 0.2,
        "allowed_rotation": 2,
        "width": 1150.4
    },{
        "index" : 13,
        "name" : "fu",
        "scale" : 20,
        "allowed_rotation": 4,
        "width": 760
    },{
        "index" : 14,
        "name" : "dagli",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 1200
    },]

class PreProccess(object):
    '''
    预处理NFP以及NFP divided函数
    '''
    def __init__(self):
        index = 2
        self.set_name = targets[index]["name"]
        self.min_angle = 360/targets[index]["allowed_rotation"]
        self.zoom = targets[index]["scale"]
        self.orientation()
        self.main()

    def orientation(self):
        fu = pd.read_csv("data/" + self.set_name + ".csv")
        _len = fu.shape[0]
        min_angle = self.min_angle
        rotation_range = [j for j in range(int(360/self.min_angle))]
        with open("data/" + self.set_name + "_orientation.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(_len):
                Poly_i = Polygon(self.normData(json.loads(fu["polygon"][i])))
                all_poly=[]
                for oi in rotation_range:
                    new_Poly_i = self.newRotation(Poly_i,oi,min_angle)
                    new_poly_i = self.getPoint(new_Poly_i)
                    all_poly.append(new_poly_i)
                if len(rotation_range) == 4:
                    ver_sym, hori_sym = 0, 0
                    if Polygon(all_poly[0]).intersection(Polygon(all_poly[2])).area == Polygon(all_poly[0]).area:
                        ver_sym = 1
                    if Polygon(all_poly[1]).intersection(Polygon(all_poly[3])).area == Polygon(all_poly[1]).area:
                        hori_sym = 1
                    all_poly.append(ver_sym)
                    all_poly.append(hori_sym)
                elif len(rotation_range) == 2:
                    ver_sym = 0
                    if Polygon(all_poly[0]).intersection(Polygon(all_poly[1])).area == Polygon(all_poly[0]).area:
                        ver_sym = 1
                    all_poly.append(ver_sym)

                writer.writerows([all_poly])

    def main(self):
        fu = pd.read_csv("data/" + self.set_name + ".csv")
        _len = fu.shape[0]
        min_angle = self.min_angle
        rotation_range = [j for j in range(int(360/self.min_angle))]
        with open("data/" + self.set_name + "_nfp.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            #for i in range(_len):
            for i in range(2,3):
                Poly_i=Polygon(self.normData(json.loads(fu["polygon"][i]))) # 固定形状
                #for j in range(_len):
                for j in range(3,4):
                    Poly_j=Polygon(self.normData(json.loads(fu["polygon"][j]))) # 移动的形状
                    for oi in rotation_range:
                        new_poly_i=self.rotation(Poly_i,oi,min_angle)
                        self.slideToOrigin(new_poly_i)
                        for oj in rotation_range:
                            print(i,j,oi,oj)
                            new_poly_j = self.rotation(Poly_j,oj,min_angle) 
                            nfp = NFP(new_poly_i,new_poly_j)
                            new_nfp = LPAssistant.deleteOnline(nfp.nfp)
                            convex_status = self.getConvexStatus(new_nfp)
                            vertical_direction = PreProccess.getVerticalDirection(convex_status,new_nfp)
                            first_pt = new_nfp[0]
                            new_NFP = Polygon(new_nfp)
                            bounds = new_NFP.bounds
                            bounds = [bounds[0]-first_pt[0],bounds[1]-first_pt[1],bounds[2]-first_pt[0],bounds[3]-first_pt[1]]
                            writer.writerows([[i,j,oi,oj,new_poly_i,new_poly_j,new_nfp,convex_status,vertical_direction,bounds]])

    def getConvexStatus(self,nfp):
        '''判断凹点还是凸点'''
        if len(nfp) == 3:
            return [1,1,1]
        convex_status = []
        for i in range(len(nfp)):
            nfp_after_del = copy.deepcopy(nfp)
            del nfp_after_del[i]
            if Polygon(nfp_after_del).contains(Point(nfp[i])):
                convex_status.append(0)
            else:
                convex_status.append(1)
        return convex_status

    @staticmethod
    def getVerticalDirection(convex_status,nfp):
        '''获得某个凹点的两个垂线'''
        target_NFP,extend_nfp = Polygon(nfp), nfp + nfp
        vertical_direction = []
        for i,status in enumerate(convex_status):
            # 如果不垂直，则需要计算垂线了
            if status == 0:
                vec1 = PreProccess.rotationDirection([extend_nfp[i][0]-extend_nfp[i-1][0],extend_nfp[i][1]-extend_nfp[i-1][1]])
                vec2 = PreProccess.rotationDirection([extend_nfp[i+1][0]-extend_nfp[i][0],extend_nfp[i+1][1]-extend_nfp[i][1]])
                vertical_direction.append([vec1,vec2])
            else:
                vertical_direction.append([[],[]])
        return vertical_direction

    @staticmethod
    def rotationDirection(vec):
        theta = math.pi/2
        new_x = vec[0] * math.cos(theta) - vec[1] * math.sin(theta)
        new_y = vec[0] * math.sin(theta) + vec[1] * math.cos(theta)
        return [new_x,new_y]

    def slideToOrigin(self,poly):
        bottom_pt,min_y = [],999999999
        for pt in poly:
            if pt[1] < min_y:
                min_y = pt[1]
                bottom_pt = [pt[0],pt[1]]
        GeoFunc.slidePoly(poly,-bottom_pt[0],-bottom_pt[1])

    def normData(self,poly):
        new_poly,num = [],self.zoom
        for pt in poly:
            new_poly.append([pt[0]*num,pt[1]*num])
        return new_poly

    def rotation(self,Poly,orientation,min_angle):
        if orientation == 0:
            return self.getPoint(Poly)
        new_Poly = affinity.rotate(Poly,min_angle*orientation)
        return self.getPoint(new_Poly)

    def newRotation(self,Poly,orientation,min_angle):
        if orientation == 0:
            return Poly
        new_Poly = affinity.rotate(Poly,min_angle*orientation)
        return new_Poly

    def getPoint(self,shapely_object):
        mapping_res = mapping(shapely_object)
        coordinates = mapping_res["coordinates"][0]
        new_poly = []
        for pt in coordinates:
            new_poly.append([pt[0],pt[1]])
        return new_poly

    def normFile(self):
        data = pd.read_csv("data/mao_orientation.csv")
        with open("data/mao_orientation.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for row in range(data.shape[0]):
                o_1 = self.normData(json.loads(data["o_1"][row]))
                o_2 = self.normData(json.loads(data["o_2"][row]))
                o_3 = self.normData(json.loads(data["o_3"][row]))
                writer.writerows([[o_0,o_1,o_2,o_3]])


class initialResult(object):
    def __init__(self,polys):
        self.polys=polys
        self.main(_type="length")
    
    def main(self,_type):
        _list=[]
        if _type=="area":
            pass
        elif _type=="length":
            _list=self.getLengthDecreaing()
        elif _type=="width":
            _list=self.getWidthDecreaing()
        elif _type=="rectangularity":
            pass
        else:
            pass
        # 重排列后的结果
        self.nfp_assistant = NFPAssistant(self.polys,store_nfp=True,get_all_nfp=False,load_history=False)
        new_list = sorted(_list, key=lambda item: item[1],reverse=True)
    
    def checkOneSeq(self,one_list):
        new_polys=[]
        for item in one_list:
            new_polys.append(self.polys[item[0]])

        packing_polys=BottomLeftFill(760,new_polys,NFPAssistant=self.nfp_assistant).polygons
        _len=LPAssistant.getLength(packing_polys)

        ratio=433200/(_len*760)

        res=[[] for i in range(len(new_polys))]
        for i,item in enumerate(one_list):
            res[one_list[i][0]]=packing_polys[i]

        return ratio,res

    def getAreaDecreaing(self):
        pass

    def getWidthDecreaing(self,polys):
        width_list=[]
        for i,poly in enumerate(self.polys):
            left_pt,right_pt = LPAssistant.getLeftPoint(poly),LPAssistant.getRightPoint(poly)
            width_list.append([i,right_pt[0]-left_pt[0]])
        return width_list

    def getLengthDecreaing(self):
        length_list=[]
        for i,poly in enumerate(self.polys):
            bottom_pt,top_pt=LPAssistant.getBottomPoint(poly),LPAssistant.getTopPoint(poly)
            length_list.append([i,top_pt[1]-bottom_pt[1]])
        return length_list

    def getRectangularityDecreaing(self,polys):
        
        pass

    def getAllSeq(self,_list):
        '''
        当前获得是全部序列
        '''
        # 初步排列
        new_list=sorted(_list, key=lambda item: item[1],reverse=True)
        # 获得全部聚类结果
        clustering,now_clustering,last_value=[],[],new_list[0][1]
        for i,item in enumerate(new_list):
            if item[1]==last_value:
                now_clustering.append(item)
            else:
                clustering.append(now_clustering)
                last_value=item[1]
                now_clustering=[item]
        clustering.append(now_clustering)
        # 获得全部序列
        all_list0=list(itertools.permutations(clustering[0]))
        all_list1=list(itertools.permutations(clustering[1]))

        n=0
        with open("/Users/sean/Documents/Projects/Data/all_list.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for permutations0 in all_list0:
                for permutations1 in all_list1:
                    print("计算第",n,"个组合")
                    one_list=list(permutations0+permutations1)+[clustering[2][0]]+[clustering[3][0]]
                    ratio,res=self.checkOneSeq(one_list)
                    writer.writerows([[n,one_list]])
                    n=n+1

class Clustering(object):
    def __init__(self):
        pass

class ReverseFunction(object):
    def __init__(self):
        new_poly = self.getReverse([[0.0, 0.0], [50.0, 150.0], [0.0, 250.0], [100.0, 200.0], [200.0, 250.0], [150.0, 100.0], [200.0, 0.0], [200.0, -150.0], [100.0, -200.0], [0.0, -150.0]])
        print(new_poly)
        # self.main()

    def main(self):
        fu = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/c_blf.csv")
        _len= fu.shape[0]
        
        for i in range(_len):
            polys=json.loads(fu["polys"][i])
            clock_polys=[]
            for poly in polys:
                new_poly=self.getReverse(poly)
                clock_polys.append(new_poly)
            with open("/Users/sean/Documents/Projects/Packing-Algorithm/record/new_c_blf.csv","a+") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([[fu["index"][i],fu["descript"][i],fu["width"][i],fu["total_area"][i],fu["overlap"][i],fu["polys_orientation"][i],clock_polys]])

    def getReverse(self,polys):
        i = len(polys)-1
        new_polys = []
        while(i >= 0):
            new_polys.append(polys[i])
            i = i - 1
        return new_polys

def showLPResult():
    fu = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/lp_result.csv")
    _len = fu.shape[0]
    for i in range(_len):
        PltFunc.addPolygon(json.loads(fu["polygon"][i]))
    PltFunc.showPlt()

def addTuplePoly(_arr,_tuple):
    """增加tuple格式的多边形，不添加最后一个"""
    for i,pt in enumerate(_tuple):
        if i == len(_tuple) - 1 :
            break
        _arr.append([pt[0],pt[1]])

def cluster():
    '''手动聚类'''
    polys = getData()
    nfp = NFP(polys[13],polys[1])
    new_nfp = LPAssistant.deleteOnline(nfp.nfp)
    # PltFunc.addPolygon(new_nfp)
    poly0 = copy.deepcopy(polys[13])
    poly1 = copy.deepcopy(polys[1])
    # poly2 = copy.deepcopy(polys[1])
    # print(new_nfp)
    GeometryAssistant.slideToPoint(poly1,[100,50])
    # GeometryAssistant.slideToPoint(poly2,[100,450])
    # PltFunc.addPolygon(poly0)
    # PltFunc.addPolygon(poly1)
    # PltFunc.addPolygon(poly2)
    final_poly = Polygon(poly0).union(Polygon(poly1))
    print(mapping(final_poly))
    _arr = []
    addTuplePoly(_arr,mapping(final_poly)["coordinates"][0])
    print(_arr)
    PltFunc.addPolygon(_arr)
    PltFunc.showPlt()
    # print(_arr)

def removeOverlap():
    _input = pd.read_csv("record/lp_initial.csv")
    polys = json.loads(_input["polys"][73])
    GeoFunc.slidePoly(polys[20],0,499.77968278349886-496.609895730301)
    GeoFunc.slidePoly(polys[5],5,0)
    PltFunc.addPolygon(polys[20])
    PltFunc.addPolygon(polys[8])
    PltFunc.addPolygon(polys[5])
    
    print(polys[8])
    print(polys[20])
    PltFunc.showPlt()
    # PltFunc.showPolys(polys)

def addBound():
    data = pd.read_csv("data/shapes0_nfp.csv")
    with open("data/shapes0_nfp.csv","a+") as csvfile:
        writer = csv.writer(csvfile)
        for row in range(data.shape[0]):
        # for row in range(500,550):
            nfp = json.loads(data["nfp"][row])
            first_pt = nfp[0]
            new_NFP = Polygon(nfp)
            bound = new_NFP.bounds
            bound = [bound[0]-first_pt[0],bound[1]-first_pt[1],bound[2]-first_pt[0],bound[3]-first_pt[1]]

            vertical_direction = PreProccess().getVerticalDirection(json.loads(data["convex_status"][row]),new_NFP)
            # vertical_direction = json.loads(data["vertical_direction"][row])
            new_vertical_direction = []
            for item in vertical_direction:
                if item == []:
                    new_vertical_direction.append([[],[]])
                else:
                    new_vertical_direction.append(item)
            writer.writerows([[data["i"][row],data["j"][row],data["oi"][row],data["oj"][row],json.loads(data["new_poly_i"][row]),json.loads(data["new_poly_j"][row]),json.loads(data["nfp"][row]),json.loads(data["convex_status"][row]),new_vertical_direction,bound]])
 

if __name__ == '__main__':
    addBound()
    # PreProccess()
