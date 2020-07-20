from tools.polygon import PltFunc,GeoFunc,NFP,getData
from sequence import BottomLeftFill
from tools.geo_assistant import GeometryAssistant,polygonQuickDecomp,Delaunay2D
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
import random
from tqdm import tqdm
from ast import literal_eval

targets_clus = [{
        "index" : 0,
        "name" : "blaz",
        "scale" : 50,
        "allowed_rotation": 2,
        "width": 750
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
    },{
        "index" : 15,
        "name" : "dighe1",
        "scale" : 10,
        "allowed_rotation": 1,
        "width": 1000
    },{
        "index" : 16,
        "name" : "dighe2",
        "scale" : 10,
        "allowed_rotation": 1,
        "width": 1000 
    }]

targets = [{
        "index" : 0,
        "name" : "albano",
        "scale" : 0.2,
        "allowed_rotation": 2,
        "width": 980
    },{
        "index" : 1,
        "name" : "blaz",
        "scale" : 50,
        "allowed_rotation": 2,
        "width": 750
    },{
        "index" : 2,
        "name" : "dagli",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 1200
    },{
        "index" : 3,
        "name" : "dighe1",
        "scale" : 10,
        "allowed_rotation": 1,
        "width": 1000
    },{
        "index" : 4,
        "name" : "dighe2",
        "scale" : 10,
        "allowed_rotation": 1,
        "width": 1000 
    },{
        "index" : 5,
        "name" : "fu",
        "scale" : 20,
        "allowed_rotation": 4,
        "width": 760
    },{
        "index" : 6,
        "name" : "jakobs1",
        "scale" : 20,
        "allowed_rotation": 4,
        "width": 800
    },{
        "index" : 7,
        "name" : "jakobs2",
        "scale" : 10,
        "allowed_rotation": 4,
        "width": 700
    },{
        "index" : 8,
        "name" : "mao",
        "scale" : 1,
        "allowed_rotation": 4,
        "width": 2550
    },{
        "index" : 9,
        "name" : "marques",
        "scale" : 10,
        "allowed_rotation": 4,
        "width": 1040
    },{
        "index" : 10,
        "name" : "shapes0",
        "scale" : 20,
        "allowed_rotation": 1,
        "width": 800
    },{
        "index" : 11,
        "name" : "shapes1",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 800
    },{
        "index" : 12,
        "name" : "shirts",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 800
    },{
        "index" : 13,
        "name" : "swim",
        "scale" : 0.2,
        "allowed_rotation": 2,
        "width": 1150.4
    },{
        "index" : 14,
        "name" : "trousers",
        "scale" : 10,
        "allowed_rotation": 2,
        "width": 790
    },{
        "index" : 15,
        "name" : "blaz_clus",
        "scale" : 50,
        "allowed_rotation": 2,
        "width": 750 
    },{
        "index" : 16,
        "name" : "dagli_clus",
        "scale" : 20,
        "allowed_rotation": 2,
        "width": 1200
    }]

class PreProccess(object):
    '''
    预处理NFP以及NFP divided函数
    '''
    def __init__(self,index):
        index = 16
        self.set_name = targets[index]["name"]
        self.min_angle = 360/targets[index]["allowed_rotation"]
        self.zoom = targets[index]["scale"]
        self.orientation()
        # self.main()

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
            for i in range(_len):
            #for i in range(2,3):
                Poly_i=Polygon(self.normData(json.loads(fu["polygon"][i]))) # 固定形状
                for j in range(_len):
                #for j in range(3,4):
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
    _input = pd.read_csv("record/best_result/fu.csv")
    polys = json.loads(_input["polys"][5])
    right = GeometryAssistant.getPolysRight(polys)
    PltFunc.addLineColor([[right,0],[right,760]])
    PltFunc.addLineColor([[0,760],[right,760]])

    # GeoFunc.slidePoly(polys[12],150-154.9999999999999,500-499.54301309142534)
    # GeoFunc.slidePoly(polys[22],0,500-494.54301309142545)
    # GeoFunc.slidePoly(polys[22],0,-3.1130634730287)
    # GeoFunc.slidePoly(polys[16],120.0-119.71600103769902,-3.1130634730287)
    # GeoFunc.slidePoly(polys[6],100.0-99.71600103769902,0)
    # GeoFunc.slidePoly(polys[8],-2.424242424242436,0)

    # GeoFunc.slidePoly(polys[3],-2.382,0)
    # PltFunc.addPolygon(polys[20])
    # PltFunc.addPolygon(polys[8])
    # PltFunc.addPolygon(polys[5])
    for i,poly in enumerate(polys):
        # print(i)
        PltFunc.addPolygon(poly)
        # PltFunc.showPlt(width=2000,height=2000)
    # print(polys[18])
    # print(polys[12])
    # print(polys[5])

    PltFunc.showPlt(width=1000,height=1000)
    # PltFunc.showPolys(polys)
    # print(polys)

def testNFP():
    data = pd.read_csv("data/dagli_nfp.csv")
    for row in range(data.shape[0]):
        nfp = json.loads(data["nfp"][row])
        GeoFunc.slidePoly(nfp,300,300)
        PltFunc.addPolygon(nfp)
        PltFunc.showPlt()

def exteriorRecord():
    data = pd.read_csv("data/fu_nfp.csv")
    # print(ast.literal_eval(res))
    with open("data/exterior/fu_nfp_exterior.csv","a+") as csvfile:
        writer = csv.writer(csvfile)
        for row in range(data.shape[0]):
            nfp = json.loads(data["nfp"][row])
            bounds = json.loads(data["bounds"][row])
            GeoFunc.slidePoly(nfp,-nfp[0][0],-nfp[0][1])
            new_NFP = Polygon(nfp)
            exterior_pts = {}
            for x in range(int(bounds[0]),int(bounds[2]+1)+1):
                for y in range(int(bounds[1]),int(bounds[3]+1)+1):
                    if new_NFP.contains(Point(x,y)) == False:
                        target_key = str(int(x)).zfill(4) + str(int(y)).zfill(4)
                        exterior_pts[target_key] = 1
            writer.writerows([[data["i"][row],data["j"][row],data["oi"][row],data["oj"][row],exterior_pts]])

def addBound(set_name):
    data = pd.read_csv("data/{}_nfp.csv".format(set_name))
    with open("data/{}_nfp.csv".format(set_name),"a+") as csvfile:
        writer = csv.writer(csvfile)
        for row in range(data.shape[0]):
        # for row in range(500,550):
            # nfp = json.loads(data["nfp"][row])
            # first_pt = nfp[0]
            # new_NFP = Polygon(nfp)
            # bound = new_NFP.bounds
            # bound = [bound[0]-first_pt[0],bound[1]-first_pt[1],bound[2]-first_pt[0],bound[3]-first_pt[1]]

            # vertical_direction = PreProccess().getVerticalDirection(json.loads(data["convex_status"][row]),new_NFP)
            # vertical_direction = json.loads(data["vertical_direction"][row])
            # new_vertical_direction = []
            # for item in vertical_direction:
            #     if item == []:
            #         new_vertical_direction.append([[],[]])
            #     else:
            #         new_vertical_direction.append(item)
            writer.writerows([[data["i"][row],data["j"][row],data["oi"][row],data["oj"][row],json.loads(data["new_poly_i"][row]),json.loads(data["new_poly_j"][row]),json.loads(data["nfp"][row]),json.loads(data["convex_status"][row]),new_vertical_direction,bound]])

def addEmptyDecom(set_name):
    data = pd.read_csv("data/{}_nfp.csv".format(set_name))
    with open("data/{}_nfp.csv".format(set_name),"a+") as csvfile:
        writer = csv.writer(csvfile)
        for row in range(data.shape[0]):
            writer.writerows([[data["i"][row],data["j"][row],data["oi"][row],data["oj"][row],json.loads(data["new_poly_i"][row]),json.loads(data["new_poly_j"][row]),json.loads(data["nfp"][row]),json.loads(data["convex_status"][row]),json.loads(data["vertical_direction"][row]),json.loads(data["bounds"][row]),[]]])

def testNFPInter():
    set_name = "fu"
    data = pd.read_csv("data/{}_nfp.csv".format(set_name))
    for k in range(100):
        i, j = random.randint(0,data.shape[0]), random.randint(0,data.shape[0])
        nfp_i, nfp_j = json.loads(data["nfp"][i]), json.loads(data["nfp"][j])
        GeoFunc.slidePoly(nfp_i,random.randint(100,400),random.randint(100,400))
        GeoFunc.slidePoly(nfp_j,random.randint(100,400),random.randint(100,400))
        nfp1_edges, nfp2_edges = GeometryAssistant.getPolyEdges(nfp_i), GeometryAssistant.getPolyEdges(nfp_j)
        inter_points, intersects = GeometryAssistant.interBetweenNFPs(nfp1_edges, nfp2_edges)
        print(intersects,inter_points)
        PltFunc.addPolygonColor(inter_points)
        PltFunc.addPolygon(nfp_i)
        PltFunc.addPolygon(nfp_j)
        PltFunc.showPlt()

def nfpDecomposition():
    '''nfp凸分解'''
    # for target in targets:
    #     data = pd.read_csv("data/{}_nfp.csv".format(target['name']))
    #     if not "bounds" in data:
    #         addBound(target['name'])
    #         print(target['name'])
    error=0
    for target in targets:
        if not 'dagli_clus' in target['name']:continue
        data = pd.read_csv("data/{}_nfp.csv".format(target['name']))
        with open("data/new/{}_nfp.csv".format(target['name']),"w+") as csvfile:
            writer = csv.writer(csvfile)
            csvfile.write('i,j,oi,oj,new_poly_i,new_poly_j,nfp,convex_status,vertical_direction,bounds,nfp_parts'+'\n')
            for row in range(data.shape[0]):
                nfp = json.loads(data["nfp"][row])
                convex_status = json.loads(data["convex_status"][row])
                first_pt = nfp[0]
                GeometryAssistant.slidePoly(nfp,-first_pt[0],-first_pt[1])
                if 0 in convex_status:
                    parts=copy.deepcopy(polygonQuickDecomp(nfp))
                    area=0
                    for p in parts:
                        poly=Polygon(p)
                        area=area+poly.area
                    if abs(Polygon(nfp).area-area)>1e-7:
                        # print('{}:{} NFP凸分解错误，面积相差{}'.format(target['name'],row,Polygon(nfp).area-area))
                        parts=[]
                        dt = Delaunay2D()
                        for pt in nfp:
                            dt.addPoint(pt)
                        triangles=copy.deepcopy(dt.exportTriangles())
                        area=0
                        for p in triangles:
                            poly=[]
                            for i in p:
                                poly.append(nfp[i])
                            parts.append(poly)
                            poly=Polygon(poly)
                            area=area+poly.area
                        if abs(Polygon(nfp).area-area)>1e-7:
                            print('{}:{} NFP凸分解错误，面积相差{}'.format(target['name'],row,Polygon(nfp).area-area))
                            # PltFunc.showPolys(parts+[nfp])
                            error=error+1
                            parts=[]
                else:
                    parts=[nfp]
                writer.writerows([[data["i"][row],data["j"][row],data["oi"][row],data["oj"][row],json.loads(data["new_poly_i"][row]),json.loads(data["new_poly_j"][row]),json.loads(data["nfp"][row]),json.loads(data["convex_status"][row]),json.loads(data["vertical_direction"][row]),json.loads(data["bounds"][row]),parts]])
    print('总错误次数{}'.format(error))            

def testInter():
    # poly1 = [[600.0, 330.6256882548512], [787.2, 330.6256882548512], [935.6793595845954, 351.3573195832804], [996.6, 347.8256882548512], [1183.8000000000002, 347.8256882548512], [1183.8, 897.2256882548512], [996.5999999999999, 897.2256882548512], [935.6793595845953, 893.6940569264219], [787.2, 914.4256882548511], [600.0, 914.4256882548511], [563.1647058823529, 871.0256882548512], [449.7999999999997, 871.0256882548512], [396.5999999999999, 761.0256882548513], [396.5999999999999, 629.2256882548512], [405.71804511278185, 622.5256882548512], [396.5999999999999, 615.8256882548511], [396.5999999999999, 484.0256882548511], [449.7999999999997, 374.0256882548511], [563.1647058823529, 374.0256882548511]]
    # poly2 = [[600.0, 694.3109894869554], [787.2, 694.3109894869554], [935.6793595845954, 715.0426208153846], [996.6, 711.5109894869554], [1183.8000000000002, 711.5109894869554], [1183.8, 1260.9109894869553], [996.5999999999999, 1260.9109894869553], [935.6793595845953, 1257.379358158526], [787.2, 1278.1109894869553], [600.0, 1278.1109894869553], [563.1647058823529, 1234.7109894869554], [449.7999999999997, 1234.7109894869554], [396.5999999999999, 1124.7109894869554], [396.5999999999999, 992.9109894869554], [405.71804511278185, 986.2109894869553], [396.5999999999999, 979.5109894869553], [396.5999999999999, 847.7109894869552], [449.7999999999997, 737.7109894869552], [563.1647058823529, 737.7109894869552]]
    # poly3 = [[1003.6017161352788, 183.9999999999999], [1090.7662002835198, 196.1704589469519], [1163.6017161352788, 183.9999999999999], [1248.5784569581476, 195.86499232728343], [1323.6017161352788, 183.9999999999999], [1480.0609978315142, 205.84583874224708], [1560.2017161352787, 201.19999999999987], [1601.4134360492003, 208.08627710643918], [1720.2017161352787, 201.19999999999987], [1720.2017161352787, 618.8], [1700.2017161352787, 692.3999999999999], [1512.8950242459432, 681.5416410498935], [1461.1013787617937, 687.608896663751], [1303.6017161352788, 709.5999999999999], [1163.6017161352788, 693.1999999999999], [1023.6017161352788, 709.5999999999999], [983.2017161352787, 661.9999999999999], [873.4017161352788, 666.1999999999999], [820.2017161352787, 556.1999999999999], [800.2017161352787, 482.5999999999999], [836.4017161352788, 455.9999999999999], [836.4017161352788, 422.90386740331485], [820.2017161352787, 410.9999999999999], [800.2017161352787, 337.39999999999986], [853.4017161352788, 227.39999999999986], [963.2017161352787, 231.5999999999999]]
    # edges1, edges2 = GeometryAssistant.getPolyEdges(poly1), GeometryAssistant.getPolyEdges(poly2)
    # inter_points, intersects = GeometryAssistant.interBetweenNFPs(edges1, edges2)
    # print(intersects, inter_points)
    # PltFunc.addLine([[1183.8000000000002, 347.8256882548512], [1183.8, 897.2256882548512]])
    # PltFunc.addLine([[996.6, 711.5109894869554], [1183.8000000000002, 711.5109894869554]])
    line1 = [[1183.8000000000002, 347.8256882548512], [1183.8, 897.2256882548512]]
    line2 = [[996.6, 711.5109894869554], [1183.8000000000002, 711.5109894869554]]
    print(GeometryAssistant.lineInter(line1,line2))
    # PltFunc.addPolygon(poly1)
    # PltFunc.addPolygon(poly2)
    # PltFunc.addPolygonColor(poly3)
    # PltFunc.showPlt(width=2500, height=2500)

def testBest():
    index = 0
    _input = pd.read_csv("record/best_result/fu.csv")
    polys = json.loads(_input["polys"][index])
    width = _input["width"][index]
    length = GeometryAssistant.getPolysRight(polys)

    PltFunc.addLineColor([[length,0],[length,width]])
    PltFunc.addLineColor([[0,width],[length,width]])
    ratio = _input["total_area"][index]/(width*length)
    print("利用比例:",ratio)
    for poly in polys:
        PltFunc.addPolygon(poly)
    PltFunc.showPlt(width=2000,height=2000)

def getKeys():
    '''对Key预处理'''
    precision=20
    for target in targets_clus:
        if not 'shapes1' in target['name']:continue
        data = pd.read_csv("data/{}_nfp.csv".format(target['name']))
        with open("data/new/{}_key.csv".format(target['name']),"w+") as csvfile:
            writer = csv.writer(csvfile)
            csvfile.write('i,j,oi,oj,grid,digital,exterior'+'\n')
            for row in tqdm(range(data.shape[0])):
                nfp = json.loads(data["nfp"][row])
                nfp_parts = json.loads(data["nfp_parts"][row])
                convex_status = json.loads(data["convex_status"][row])
                first_pt = nfp[0]
                GeometryAssistant.slidePoly(nfp,-first_pt[0],-first_pt[1])
                grid=dict()
                exterior=dict()
                digital=dict()
                for x in range(-500,500,precision):
                    for y in range(-500,500,precision):
                        if not GeometryAssistant.boundsContain(Polygon(nfp).bounds,[x,y]):
                            continue
                        grid_key = str(int(x/precision)).zfill(5) + str(int(y/precision)).zfill(5)
                        further_calc=False
                        if not Polygon(nfp).contains(Point([x,y])):
                            dist=Point([x,y]).distance(Polygon(nfp))
                            if dist>15:
                                grid[grid_key]=-1
                            else:   further_calc=True
                        else:
                            depth=GeometryAssistant.getPtNFPPD([x,y], convex_status, nfp, 0.000001)
                            if depth>15:
                                grid[grid_key]=depth
                            else:   further_calc=True
                        if further_calc:
                            for m in range(x-10,x+10):
                                for n in range(y-10,y+10):
                                    digital_key = str(int(m)).zfill(6) + str(int(n)).zfill(6)
                                    if digital_key in exterior.keys() or digital_key in digital.keys():
                                        continue
                                    if not Polygon(nfp).contains(Point([m,n])):
                                        exterior[digital_key]=1
                                    else:
                                        depth=GeometryAssistant.getPtNFPPD([m,n], convex_status, nfp, 0.000001)
                                        digital[digital_key]=depth
                writer.writerows([[data["i"][row],data["j"][row],data["oi"][row],data["oj"][row],json.dumps(grid),json.dumps(digital),json.dumps(exterior)]])   

if __name__ == '__main__':
    removeOverlap()
    # testBest()
    # addEmptyDecom("swim")
    # testInter()
    # testNFP()
    # testNFPInter()
    # print(str(int(-1005/10)*10).zfill(5))
    # addBound()
    # PreProccess(12)
    # nfpDecomposition()
    # removeOverlap()
    getKeys()