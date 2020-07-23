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

def getKeys(target):
    '''对Key预处理'''
    precision=10
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
                        if dist>7.5:
                            grid[grid_key]=-1
                        else:   further_calc=True
                    else:
                        depth=GeometryAssistant.getPtNFPPD([x,y], convex_status, nfp, 0.000001)
                        if depth>7.5:
                            grid[grid_key]=depth
                        else:   further_calc=True
                    if further_calc:
                        for m in range(x-5,x+5):
                            for n in range(y-5,y+5):
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
    # removeOverlap()
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
    # for target in targets:
    #     if target['name'] in ['shapes0']:
    #         getKeys(target)
    polys=[[[641.45321901181, 85.66455174863367], [642.45321901181, 149.8645517486337], [641.2532190118101, 214.06455174863368], [611.2532190118101, 217.66455174863367], [589.65321901181, 223.8645517486337], [556.8532190118101, 233.26455174863366], [532.05321901181, 239.66455174863367], [505.45321901181006, 245.8645517486337], [469.85321901181004, 253.06455174863368], [443.85321901181004, 222.06455174863368], [426.45321901181006, 209.66455174863367], [399.2532190118101, 196.26455174863366], [377.65321901181005, 191.06455174863368], [345.2532190118101, 190.8645517486337], [322.05321901181003, 193.66455174863367], [294.05321901181003, 198.46455174863368], [297.65321901181005, 173.66455174863367], [299.85321901181004, 149.06455174863368], [297.85321901181004, 124.46455174863368], [294.25321901181, 99.66455174863367], [322.25321901181, 104.66455174863367], [345.45321901181006, 107.46455174863367], [377.85321901181004, 107.46455174863367], [399.45321901181006, 102.26455174863366], [426.65321901181005, 89.06455174863368], [444.2532190118101, 76.66455174863367], [470.2532190118101, 45.86455174863366], [505.8532190118101, 53.264551748633664], [532.45321901181, 59.46455174863365], [557.2532190118101, 66.06455174863368], [590.05321901181, 75.66455174863367], [611.65321901181, 81.86455174863366], [641.45321901181, 85.66455174863367]], [[1.0, 577.5345702319813], [0.0, 513.3345702319813], [1.2, 449.13457023198134], [31.200000000000003, 445.5345702319813], [52.800000000000004, 439.3345702319813], [85.60000000000001, 429.9345702319813], [110.4, 423.5345702319813], [137.0, 417.3345702319813], [172.60000000000002, 410.13457023198134], [198.60000000000002, 441.13457023198134], [216.0, 453.5345702319813], [243.20000000000002, 466.9345702319813], [264.8, 472.13457023198134], [297.2, 472.33457023198133], [320.40000000000003, 469.5345702319813], [348.40000000000003, 464.7345702319813], [344.8, 489.5345702319813], [342.6, 514.1345702319813], [344.6, 538.7345702319814], [348.20000000000005, 563.5345702319813], [320.20000000000005, 558.5345702319813], [297.0, 555.7345702319814], [264.6, 555.7345702319814], [243.0, 560.9345702319813], [215.8, 574.1345702319813], [198.20000000000002, 586.5345702319813], [172.20000000000002, 617.3345702319813], [136.6, 609.9345702319813], [110.0, 603.7345702319814], [85.2, 597.1345702319813], [52.400000000000006, 587.5345702319813], [30.8, 581.3345702319813], [1.0, 577.5345702319813]], [[347.40000000000003, 889.2177500210213], [348.40000000000003, 953.4177500210213], [347.20000000000005, 1017.6177500210213], [317.20000000000005, 1021.2177500210213], [295.6, 1027.4177500210212], [262.8, 1036.8177500210213], [238.00000000000003, 1043.2177500210214], [211.40000000000003, 1049.4177500210212], [175.8, 1056.6177500210213], [149.8, 1025.6177500210213], [132.40000000000003, 1013.2177500210213], [105.20000000000002, 999.8177500210213], [83.60000000000002, 994.6177500210213], [51.200000000000045, 994.4177500210213], [28.0, 997.2177500210213], [0.0, 1002.0177500210214], [3.6000000000000227, 977.2177500210213], [5.800000000000011, 952.6177500210213], [3.8000000000000114, 928.0177500210214], [0.19999999999998863, 903.2177500210213], [28.19999999999999, 908.2177500210213], [51.400000000000034, 911.0177500210214], [83.80000000000001, 911.0177500210214], [105.40000000000003, 905.8177500210213], [132.60000000000002, 892.6177500210213], [150.20000000000002, 880.2177500210213], [176.20000000000002, 849.4177500210213], [211.80000000000004, 856.8177500210213], [238.40000000000003, 863.0177500210214], [263.20000000000005, 869.6177500210213], [296.0, 879.2177500210213], [317.6, 885.4177500210213], [347.40000000000003, 889.2177500210213]], [[656.219232546973, 614.9842719024593], [657.819232546973, 585.3842719024593], [661.611685346973, 555.0446493024593], [669.907927946973, 522.4978517024593], [692.019232546973, 525.1842719024594], [713.819232546973, 525.7842719024593], [736.419232546973, 522.3842719024593], [765.819232546973, 515.5842719024594], [809.022374546973, 505.8719377024593], [817.219232546973, 544.9842719024593], [833.619232546973, 575.7842719024593], [807.219232546973, 579.7842719024593], [787.219232546973, 583.9842719024593], [760.819232546973, 590.7842719024593], [710.619232546973, 604.5842719024594], [679.709893746973, 612.6088117024593], [656.219232546973, 614.9842719024593]], [[530.6761896183774, 454.11278362344655], [529.0761896183774, 483.7127836234465], [525.2837368183774, 514.0524062234465], [516.9874942183774, 546.5992038234465], [494.87618961837734, 543.9127836234466], [473.0761896183774, 543.3127836234465], [450.47618961837736, 546.7127836234465], [421.0761896183774, 553.5127836234466], [377.87304761837737, 563.2251178234466], [369.6761896183774, 524.1127836234466], [353.2761896183774, 493.31278362344653], [379.6761896183774, 489.31278362344653], [399.6761896183774, 485.11278362344655], [426.0761896183774, 478.31278362344653], [476.2761896183774, 464.5127836234466], [507.1855284183774, 456.48824382344657], [530.6761896183774, 454.11278362344655]], [[731.6713741519897, 246.03734521786453], [733.2713741519897, 216.43734521786453], [737.0638269519897, 186.09772261786452], [745.3600695519897, 153.55092501786453], [767.4713741519896, 156.23734521786452], [789.2713741519897, 156.83734521786454], [811.8713741519897, 153.4373452178645], [841.2713741519897, 146.63734521786452], [884.4745161519896, 136.9250110178645], [892.6713741519897, 176.03734521786453], [909.0713741519896, 206.83734521786454], [882.6713741519897, 210.83734521786454], [862.6713741519897, 215.03734521786453], [836.2713741519897, 221.83734521786454], [786.0713741519896, 235.63734521786452], [755.1620353519896, 243.66188501786453], [731.6713741519897, 246.03734521786453]], [[1181.380931040064, 5.684341886080802e-14], [1179.7809310400642, 29.60000000000005], [1175.988478240064, 59.93962260000006], [1167.6922356400642, 92.48642020000005], [1145.5809310400641, 89.80000000000007], [1123.7809310400642, 89.20000000000006], [1101.180931040064, 92.60000000000007], [1071.7809310400642, 99.40000000000006], [1028.577789040064, 109.11233420000008], [1020.3809310400641, 70.00000000000006], [1003.9809310400641, 39.20000000000006], [1030.380931040064, 35.20000000000006], [1050.380931040064, 31.000000000000057], [1076.7809310400642, 24.20000000000006], [1126.9809310400642, 10.400000000000063], [1157.890269840064, 2.375460200000063], [1181.380931040064, 5.684341886080802e-14]], [[921.0647772187468, 356.5104573255771], [922.6647772187468, 326.91045732557706], [926.4572300187468, 296.5708347255771], [934.7534726187469, 264.0240371255771], [956.8647772187468, 266.71045732557707], [978.6647772187468, 267.3104573255771], [1001.2647772187469, 263.91045732557706], [1030.6647772187468, 257.11045732557704], [1073.867919218747, 247.39812312557706], [1082.0647772187467, 286.5104573255771], [1098.4647772187468, 317.3104573255771], [1072.0647772187467, 321.3104573255771], [1052.0647772187467, 325.5104573255771], [1025.6647772187468, 332.3104573255771], [975.4647772187468, 346.1104573255771], [944.5554384187468, 354.13499712557706], [921.0647772187468, 356.5104573255771]], [[642.1509135004534, 130.45653791953507], [643.7509135004534, 100.85653791953507], [647.5433663004534, 70.51691531953506], [655.8396089004534, 37.97011771953507], [677.9509135004533, 40.656537919535054], [699.7509135004534, 41.25653791953506], [722.3509135004534, 37.85653791953506], [751.7509135004534, 31.05653791953506], [794.9540555004534, 21.344203719535045], [803.1509135004534, 60.456537919535066], [819.5509135004534, 91.25653791953506], [793.1509135004534, 95.25653791953506], [773.1509135004534, 99.45653791953507], [746.7509135004534, 106.25653791953506], [696.5509135004534, 120.05653791953506], [665.6415747004534, 128.08107771953507], [642.1509135004534, 130.45653791953507]], [[258.40000000000003, 708.0930411391312], [292.20000000000005, 716.6930411391312], [321.40000000000003, 723.4930411391312], [365.5692308, 732.6073269391312], [387.92903220000005, 742.9886633391312], [333.0, 747.8930411391311], [308.20000000000005, 751.0930411391312], [269.6, 759.6930411391312], [242.00000000000003, 771.2930411391312], [200.60000000000002, 795.0930411391312], [173.8, 788.8930411391311], [131.0256318, 779.8628967391312], [96.09756100000001, 768.1014035391312], [73.4, 757.2930411391312], [48.0, 743.4930411391312], [3.6000000000000227, 722.6930411391312], [1.6000000000000227, 697.2930411391312], [0.0, 672.4930411391312], [38.60000000000002, 666.4930411391312], [68.20000000000002, 662.0930411391312], [90.60000000000002, 659.2930411391312], [121.80000000000001, 657.0930411391312], [155.09473680000002, 657.0930411391312], [190.40000000000003, 662.2930411391312], [226.4094334, 670.2566659391312], [258.40000000000003, 708.0930411391312]], [[298.4845632483787, 1099.221356282178], [264.6845632483787, 1090.621356282178], [235.48456324837872, 1083.821356282178], [191.31533244837874, 1074.707070482178], [168.9555310483787, 1064.325734082178], [223.88456324837873, 1059.421356282178], [248.6845632483787, 1056.221356282178], [287.28456324837873, 1047.621356282178], [314.88456324837875, 1036.021356282178], [356.28456324837873, 1012.221356282178], [383.08456324837874, 1018.4213562821781], [425.8589314483787, 1027.451500682178], [460.7870022483787, 1039.2129938821781], [483.4845632483788, 1050.021356282178], [508.88456324837875, 1063.821356282178], [553.2845632483787, 1084.621356282178], [555.2845632483787, 1110.021356282178], [556.8845632483788, 1134.821356282178], [518.2845632483787, 1140.821356282178], [488.6845632483787, 1145.221356282178], [466.28456324837873, 1148.021356282178], [435.08456324837874, 1150.221356282178], [401.78982644837873, 1150.221356282178], [366.4845632483787, 1145.021356282178], [330.47512984837874, 1137.057731482178], [298.4845632483787, 1099.221356282178]], [[356.2546313452421, 242.98217176687876], [390.0546313452421, 251.58217176687876], [419.2546313452421, 258.38217176687874], [463.4238621452421, 267.4964575668788], [485.7836635452421, 277.87779396687876], [430.85463134524207, 282.7821717668788], [406.0546313452421, 285.98217176687876], [367.4546313452421, 294.5821717668788], [339.85463134524207, 306.18217176687875], [298.4546313452421, 329.98217176687876], [271.6546313452421, 323.7821717668788], [228.88026314524208, 314.7520273668788], [193.95219234524208, 302.9905341668788], [171.25463134524207, 292.18217176687875], [145.85463134524207, 278.38217176687874], [101.45463134524209, 257.5821717668788], [99.45463134524209, 232.18217176687875], [97.85463134524207, 207.38217176687877], [136.4546313452421, 201.38217176687877], [166.0546313452421, 196.98217176687876], [188.4546313452421, 194.18217176687875], [219.65463134524208, 191.98217176687876], [252.9493681452421, 191.98217176687876], [288.2546313452421, 197.18217176687875], [324.2640647452421, 205.14579656687874], [356.2546313452421, 242.98217176687876]], [[258.40000000000026, 51.0], [292.2000000000003, 59.599999999999994], [321.40000000000026, 66.4], [365.56923080000024, 75.51428580000001], [387.9290322000003, 85.8956222], [333.0000000000002, 90.80000000000001], [308.2000000000003, 94.0], [269.60000000000025, 102.60000000000001], [242.00000000000026, 114.2], [200.60000000000025, 138.0], [173.80000000000024, 131.8], [131.02563180000024, 122.76985560000001], [96.09756100000024, 111.00836240000001], [73.40000000000023, 100.2], [48.00000000000023, 86.4], [3.60000000000025, 65.6], [1.6000000000002501, 40.2], [2.2737367544323206e-13, 15.400000000000006], [38.60000000000025, 9.400000000000006], [68.20000000000024, 5.0], [90.60000000000025, 2.200000000000003], [121.80000000000024, 0.0], [155.09473680000025, 0.0], [190.40000000000026, 5.200000000000003], [226.40943340000024, 13.163624799999994], [258.40000000000026, 51.0]], [[218.65116242200628, 661.0806843037624], [184.85116242200627, 652.4806843037624], [155.65116242200628, 645.6806843037624], [111.4819316220063, 636.5663985037623], [89.12213022200626, 626.1850621037623], [144.05116242200629, 621.2806843037624], [168.85116242200627, 618.0806843037624], [207.4511624220063, 609.4806843037624], [235.05116242200629, 597.8806843037623], [276.4511624220063, 574.0806843037624], [303.2511624220063, 580.2806843037624], [346.0255306220063, 589.3108287037624], [380.9536014220063, 601.0723219037624], [403.65116242200634, 611.8806843037623], [429.0511624220063, 625.6806843037624], [473.4511624220063, 646.4806843037624], [475.4511624220063, 671.8806843037623], [477.0511624220063, 696.6806843037624], [438.4511624220063, 702.6806843037624], [408.85116242200627, 707.0806843037624], [386.4511624220063, 709.8806843037623], [355.2511624220063, 712.0806843037624], [321.9564256220063, 712.0806843037624], [286.6511624220063, 706.8806843037623], [250.6417290220063, 698.9170595037624], [218.65116242200628, 661.0806843037624]], [[223.64616537827305, 429.69168306231074], [189.84616537827304, 421.0916830623107], [160.64616537827305, 414.29168306231077], [116.47693457827307, 405.1773972623107], [94.11713317827304, 394.79606086231075], [149.04616537827306, 389.89168306231073], [173.84616537827304, 386.69168306231074], [212.44616537827307, 378.0916830623107], [240.04616537827306, 366.49168306231076], [281.44616537827307, 342.69168306231074], [308.2461653782731, 348.89168306231073], [351.0205335782731, 357.9218274623107], [385.9486043782731, 369.68332066231073], [408.64616537827305, 380.49168306231076], [434.0461653782731, 394.29168306231077], [478.44616537827307, 415.0916830623107], [480.44616537827307, 440.49168306231076], [482.0461653782731, 465.29168306231077], [443.44616537827307, 471.29168306231077], [413.8461653782731, 475.69168306231074], [391.44616537827307, 478.49168306231076], [360.2461653782731, 480.69168306231074], [326.95142857827307, 480.69168306231074], [291.64616537827305, 475.49168306231076], [255.63673197827308, 467.52805826231076], [223.64616537827305, 429.69168306231074]], [[960.7555233982819, 103.80775963070046], [959.155523398282, 138.20775963070045], [898.9555233982819, 137.00775963070046], [863.9555233982819, 136.80775963070045], [837.5555233982819, 137.20775963070045], [817.155523398282, 138.20775963070045], [785.5555233982819, 142.60775963070046], [754.5555233982819, 153.40775963070047], [690.4645429982819, 126.02711923070046], [718.9555233982819, 115.20775963070045], [750.9613985982819, 105.17201903070045], [778.9555233982819, 100.60775963070046], [803.7555233982819, 98.60775963070046], [827.155523398282, 98.00775963070046], [858.5555233982819, 98.60775963070046], [918.155523398282, 101.00775963070046], [960.7555233982819, 103.80775963070046]], [[556.4379339092767, 1144.423887072873], [558.0379339092767, 1110.0238870728729], [618.2379339092766, 1111.223887072873], [653.2379339092768, 1111.423887072873], [679.6379339092767, 1111.0238870728729], [700.0379339092767, 1110.0238870728729], [731.6379339092767, 1105.623887072873], [762.6379339092767, 1094.823887072873], [826.7289143092767, 1122.204527472873], [798.2379339092768, 1133.0238870728729], [766.2320587092768, 1143.059627672873], [738.2379339092768, 1147.623887072873], [713.4379339092767, 1149.623887072873], [690.0379339092767, 1150.223887072873], [658.6379339092767, 1149.623887072873], [599.0379339092767, 1147.223887072873], [556.4379339092767, 1144.423887072873]], [[381.8456821567569, 74.71642053741375], [383.4456821567569, 40.316420537413755], [443.6456821567569, 41.51642053741375], [478.6456821567569, 41.716420537413754], [505.04568215675687, 41.316420537413755], [525.4456821567569, 40.316420537413755], [557.0456821567569, 35.91642053741375], [588.0456821567569, 25.116420537413752], [652.1366625567568, 52.49706093741375], [623.645682156757, 63.316420537413755], [591.639806956757, 73.35216113741376], [563.645682156757, 77.91642053741376], [538.8456821567569, 79.91642053741376], [515.4456821567569, 80.51642053741375], [484.04568215675687, 79.91642053741376], [424.4456821567569, 77.51642053741375], [381.8456821567569, 74.71642053741375]], [[1181.450583673476, 823.6901277528177], [1179.850583673476, 858.0901277528177], [1119.650583673476, 856.8901277528176], [1084.650583673476, 856.6901277528177], [1058.250583673476, 857.0901277528177], [1037.850583673476, 858.0901277528177], [1006.2505836734761, 862.4901277528177], [975.2505836734761, 873.2901277528176], [911.159603273476, 845.9094873528177], [939.650583673476, 835.0901277528177], [971.656458873476, 825.0543871528176], [999.650583673476, 820.4901277528177], [1024.450583673476, 818.4901277528177], [1047.850583673476, 817.8901277528176], [1079.250583673476, 818.4901277528177], [1138.850583673476, 820.8901277528176], [1181.450583673476, 823.6901277528177]], [[510.0631917450464, 6.748895557595766], [508.4631917450464, 41.148895557595765], [448.26319174504636, 39.94889555759577], [413.26319174504636, 39.748895557595766], [386.8631917450464, 40.148895557595765], [366.4631917450464, 41.148895557595765], [334.8631917450464, 45.54889555759577], [303.8631917450464, 56.34889555759577], [239.77221134504637, 28.96825515759577], [268.26319174504636, 18.148895557595765], [300.26906694504635, 8.113154957595768], [328.26319174504636, 3.5488955575957633], [353.0631917450464, 1.5488955575957633], [376.4631917450464, 0.9488955575957689], [407.8631917450464, 1.5488955575957633], [467.4631917450464, 3.948895557595769], [510.0631917450464, 6.748895557595766]], [[1072.4654989519897, 213.67308581786455], [1070.8654989519896, 248.07308581786455], [1010.6654989519897, 246.87308581786456], [975.6654989519897, 246.67308581786455], [949.2654989519897, 247.07308581786455], [928.8654989519897, 248.07308581786455], [897.2654989519897, 252.47308581786456], [866.2654989519897, 263.27308581786457], [802.1745185519896, 235.89244541786456], [830.6654989519897, 225.07308581786455], [862.6713741519897, 215.03734521786456], [890.6654989519897, 210.47308581786456], [915.4654989519897, 208.47308581786456], [938.8654989519897, 207.87308581786456], [970.2654989519897, 208.47308581786456], [1029.8654989519896, 210.87308581786456], [1072.4654989519897, 213.67308581786455]], [[1083.7123378734761, 476.2966240551688], [1090.504337873476, 447.7136240551688], [1099.1123378734762, 422.6966240551688], [1101.512337873476, 398.6966240551688], [918.5123378734761, 388.4966240551688], [918.9123378734762, 356.6966240551688], [1091.512337873476, 354.8966240551688], [1117.512337873476, 346.2966240551688], [1125.512337873476, 316.2966240551688], [1133.9123378734762, 288.2966240551688], [1157.9123378734762, 291.4966240551688], [1154.512337873476, 331.09662405516883], [1153.312337873476, 354.2966240551688], [1153.7123378734761, 393.8966240551688], [1155.7123378734761, 423.4966240551688], [1160.1123378734762, 457.4966240551688], [1181.450583673476, 522.7191868551688], [1154.1123378734762, 487.09662405516883], [1127.1123378734762, 476.2966240551688], [1083.7123378734761, 476.2966240551688]], [[1083.7123378734761, 277.4788460596353], [1090.504337873476, 248.89584605963532], [1099.1123378734762, 223.87884605963532], [1101.512337873476, 199.8788460596353], [918.5123378734761, 189.6788460596353], [918.9123378734762, 157.8788460596353], [1091.512337873476, 156.0788460596353], [1117.512337873476, 147.47884605963532], [1125.512337873476, 117.47884605963532], [1133.9123378734762, 89.47884605963532], [1157.9123378734762, 92.6788460596353], [1154.512337873476, 132.2788460596353], [1153.312337873476, 155.47884605963532], [1153.7123378734761, 195.0788460596353], [1155.7123378734761, 224.6788460596353], [1160.1123378734762, 258.6788460596353], [1181.450583673476, 323.9014088596353], [1154.1123378734762, 288.2788460596353], [1127.1123378734762, 277.4788460596353], [1083.7123378734761, 277.4788460596353]], [[422.2179511030781, 826.3191413294712], [415.4259511030781, 854.9021413294712], [406.8179511030781, 879.9191413294711], [404.4179511030781, 903.9191413294712], [587.4179511030782, 914.1191413294712], [587.0179511030781, 945.9191413294712], [414.4179511030781, 947.7191413294712], [388.41795110307805, 956.3191413294712], [380.41795110307805, 986.3191413294712], [372.0179511030781, 1014.3191413294712], [348.0179511030781, 1011.1191413294712], [351.4179511030781, 971.5191413294713], [352.6179511030781, 948.3191413294712], [352.2179511030781, 908.7191413294712], [350.2179511030781, 879.1191413294712], [345.8179511030781, 845.1191413294712], [324.47970530307805, 779.8965785294712], [351.8179511030781, 815.5191413294712], [378.8179511030781, 826.3191413294712], [422.2179511030781, 826.3191413294712]], [[481.3567300189506, 683.4645695712969], [488.1487300189506, 654.881569571297], [496.7567300189506, 629.8645695712969], [499.1567300189506, 605.8645695712969], [316.1567300189506, 595.6645695712969], [316.5567300189506, 563.8645695712969], [489.1567300189506, 562.0645695712969], [515.1567300189506, 553.4645695712969], [523.1567300189506, 523.4645695712969], [531.5567300189506, 495.4645695712969], [555.5567300189506, 498.6645695712969], [552.1567300189506, 538.2645695712969], [550.9567300189506, 561.4645695712969], [551.3567300189507, 601.0645695712969], [553.3567300189507, 630.664569571297], [557.7567300189506, 664.664569571297], [579.0949758189506, 729.887132371297], [551.7567300189506, 694.2645695712969], [524.7567300189506, 683.4645695712969], [481.3567300189506, 683.4645695712969]], [[97.73824580000002, 233.96145325465133], [90.94624580000001, 262.54445325465133], [82.33824580000001, 287.5614532546513], [79.93824580000002, 311.5614532546513], [262.9382458, 321.7614532546513], [262.5382458, 353.5614532546513], [89.93824580000002, 355.36145325465134], [63.938245800000004, 363.96145325465136], [55.938245800000004, 393.96145325465136], [47.53824580000001, 421.96145325465136], [23.5382458, 418.7614532546513], [26.938245800000033, 379.16145325465135], [28.13824580000002, 355.96145325465136], [27.738245800000016, 316.36145325465134], [25.738245800000016, 286.7614532546513], [21.33824580000001, 252.76145325465131], [0.0, 187.53889045465132], [27.33824580000001, 223.16145325465132], [54.33824580000001, 233.96145325465133], [97.73824580000002, 233.96145325465133]], [[482.3923154201827, 438.58031473828976], [489.18431542018266, 409.9973147382898], [497.79231542018266, 384.98031473828974], [500.1923154201827, 360.98031473828974], [317.1923154201827, 350.78031473828975], [317.5923154201827, 318.98031473828974], [490.1923154201827, 317.1803147382898], [516.1923154201827, 308.58031473828976], [524.1923154201827, 278.58031473828976], [532.5923154201827, 250.58031473828976], [556.5923154201827, 253.78031473828975], [553.1923154201827, 293.3803147382897], [551.9923154201826, 316.58031473828976], [552.3923154201827, 356.1803147382898], [554.3923154201827, 385.78031473828975], [558.7923154201827, 419.78031473828975], [580.1305612201827, 485.00287753828974], [552.7923154201827, 449.38031473828977], [525.7923154201827, 438.58031473828976], [482.3923154201827, 438.58031473828976]], [[733.6684412845952, 1100.172409266811], [707.7084412845952, 1105.941298266811], [672.2915182845952, 1094.135657266811], [634.6684412845952, 1076.727964866811], [610.4684412845952, 1064.3279648668108], [586.2684412845953, 1051.127964866811], [562.6684412845952, 1040.727964866811], [541.8684412845953, 1036.127964866811], [519.4684412845952, 1037.5279648668109], [478.66844128459525, 1047.727964866811], [481.66844128459525, 1019.927964866811], [483.66844128459525, 998.927964866811], [481.66844128459525, 977.5279648668109], [478.66844128459525, 949.7279648668109], [519.4684412845952, 959.927964866811], [541.8684412845953, 961.3279648668109], [562.6684412845952, 956.7279648668109], [586.2684412845953, 946.3279648668109], [610.4684412845952, 933.1279648668109], [634.6684412845952, 920.7279648668109], [672.2915182845952, 903.3202724668109], [707.7084412845952, 891.5146314668109], [733.6684412845952, 897.2835204668108], [733.6684412845952, 1100.172409266811]], [[833.179232546973, 788.4420497024594], [807.219232546973, 794.2109387024593], [771.802309546973, 782.4052977024593], [734.179232546973, 764.9976053024593], [709.979232546973, 752.5976053024593], [685.779232546973, 739.3976053024593], [662.179232546973, 728.9976053024593], [641.3792325469731, 724.3976053024593], [618.979232546973, 725.7976053024593], [578.179232546973, 735.9976053024593], [581.179232546973, 708.1976053024594], [583.179232546973, 687.1976053024594], [581.179232546973, 665.7976053024593], [578.179232546973, 637.9976053024593], [618.979232546973, 648.1976053024594], [641.3792325469731, 649.5976053024593], [662.179232546973, 644.9976053024593], [685.779232546973, 634.5976053024593], [709.979232546973, 621.3976053024593], [734.179232546973, 608.9976053024593], [771.802309546973, 591.5899129024593], [807.219232546973, 579.7842719024593], [833.179232546973, 585.5531609024592], [833.179232546973, 788.4420497024594]], [[456.9290826814309, 701.1522576655701], [482.88908268143086, 695.3833686655701], [518.3060056814309, 707.1890096655701], [555.9290826814308, 724.5967020655701], [580.1290826814309, 736.99670206557], [604.3290826814309, 750.1967020655701], [627.9290826814308, 760.5967020655701], [648.7290826814309, 765.1967020655701], [671.1290826814309, 763.7967020655701], [711.9290826814308, 753.5967020655701], [708.9290826814308, 781.39670206557], [706.9290826814308, 802.39670206557], [708.9290826814308, 823.7967020655701], [711.9290826814308, 851.5967020655701], [671.1290826814309, 841.39670206557], [648.7290826814309, 839.99670206557], [627.9290826814308, 844.5967020655701], [604.3290826814309, 854.99670206557], [580.1290826814309, 868.1967020655701], [555.9290826814308, 880.5967020655701], [518.3060056814309, 898.00439446557], [482.88908268143086, 909.8100354655701], [456.9290826814309, 904.0411464655701], [456.9290826814309, 701.1522576655701]], [[923.3376637990575, 533.2880469403167], [949.0598859990575, 548.9921405403167], [969.6598859990576, 557.3921405403166], [990.2598859990576, 560.3921405403166], [1022.8598859990575, 557.1921405403167], [1046.2598859990576, 552.5921405403167], [1086.2598859990576, 543.5921405403167], [1094.0598859990575, 577.1921405403167], [1103.050583673476, 574.4307119688881], [1124.250583673476, 567.0307119688881], [1149.6505836734761, 550.8307119688882], [1181.450583673476, 550.6307119688881], [1176.8505836734762, 583.8307119688882], [1175.050583673476, 616.4307119688881], [1174.250583673476, 650.0307119688881], [1173.450583673476, 674.6307119688881], [1174.250583673476, 699.2307119688882], [1175.050583673476, 732.8307119688882], [1176.8505836734762, 765.4307119688881], [1181.450583673476, 798.6307119688881], [1149.6505836734761, 798.4307119688881], [1124.250583673476, 782.2307119688882], [1103.050583673476, 774.8307119688882], [1094.4598859990574, 772.1921405403167], [1086.8598859990575, 805.7921405403167], [1046.8598859990575, 796.9921405403167], [1023.2598859990576, 792.3921405403167], [990.6598859990576, 789.3921405403167], [970.0598859990575, 792.3921405403167], [949.6598859990576, 800.9921405403167], [924.3422389990576, 816.4492303403167], [886.9755485990574, 832.5759071403168], [824.4598859990574, 818.5921405403167], [834.2598859990576, 776.7921405403167], [838.6598859990575, 747.3921405403167], [840.0598859990575, 703.1921405403167], [839.8598859990575, 675.1921405403167], [839.8598859990575, 647.1921405403167], [839.0598859990575, 617.5921405403167], [836.8598859990575, 590.3921405403166], [823.8598859990575, 531.7921405403167], [885.0903845990574, 517.1839293403167], [923.3376637990575, 533.2880469403167]], [[670.1620185141817, 238.79345465924484], [695.8842407141817, 254.49754825924487], [716.4842407141817, 262.89754825924484], [737.0842407141818, 265.89754825924484], [769.6842407141817, 262.69754825924485], [793.0842407141818, 258.09754825924483], [833.0842407141818, 249.09754825924486], [840.8842407141817, 282.69754825924485], [849.8749383886003, 279.9361196878163], [871.0749383886002, 272.53611968781627], [896.4749383886003, 256.3361196878163], [928.2749383886003, 256.1361196878163], [923.6749383886003, 289.3361196878163], [921.8749383886003, 321.9361196878163], [921.0749383886002, 355.53611968781627], [920.2749383886003, 380.1361196878163], [921.0749383886002, 404.7361196878163], [921.8749383886003, 438.3361196878163], [923.6749383886003, 470.9361196878163], [928.2749383886003, 504.1361196878163], [896.4749383886003, 503.9361196878163], [871.0749383886002, 487.7361196878163], [849.8749383886003, 480.3361196878163], [841.2842407141817, 477.69754825924485], [833.6842407141817, 511.2975482592449], [793.6842407141817, 502.4975482592448], [770.0842407141818, 497.8975482592449], [737.4842407141817, 494.8975482592449], [716.8842407141817, 497.8975482592449], [696.4842407141817, 506.4975482592448], [671.1665937141818, 521.9546380592449], [633.7999033141816, 538.0813148592449], [571.2842407141816, 524.0975482592448], [581.0842407141818, 482.2975482592449], [585.4842407141816, 452.8975482592449], [586.8842407141817, 408.69754825924485], [586.6842407141817, 380.69754825924485], [586.6842407141817, 352.69754825924485], [585.8842407141817, 323.09754825924483], [583.6842407141817, 295.89754825924484], [570.6842407141817, 237.29754825924485], [631.9147393141816, 222.68933705924485], [670.1620185141817, 238.79345465924484]], [[923.3376637990575, 851.1121398], [949.0598859990575, 866.8162334], [969.6598859990576, 875.2162334], [990.2598859990576, 878.2162334], [1022.8598859990575, 875.0162334], [1046.2598859990576, 870.4162334], [1086.2598859990576, 861.4162334], [1094.0598859990575, 895.0162334], [1103.050583673476, 892.2548048285714], [1124.250583673476, 884.8548048285714], [1149.6505836734761, 868.6548048285715], [1181.450583673476, 868.4548048285715], [1176.8505836734762, 901.6548048285715], [1175.050583673476, 934.2548048285714], [1174.250583673476, 967.8548048285714], [1173.450583673476, 992.4548048285715], [1174.250583673476, 1017.0548048285715], [1175.050583673476, 1050.6548048285715], [1176.8505836734762, 1083.2548048285714], [1181.450583673476, 1116.4548048285715], [1149.6505836734761, 1116.2548048285714], [1124.250583673476, 1100.0548048285714], [1103.050583673476, 1092.6548048285715], [1094.4598859990574, 1090.0162334000001], [1086.8598859990575, 1123.6162334], [1046.8598859990575, 1114.8162334], [1023.2598859990576, 1110.2162334], [990.6598859990576, 1107.2162334], [970.0598859990575, 1110.2162334], [949.6598859990576, 1118.8162334], [924.3422389990576, 1134.2733232], [886.9755485990574, 1150.4], [824.4598859990574, 1136.4162334], [834.2598859990576, 1094.6162334], [838.6598859990575, 1065.2162334], [840.0598859990575, 1021.0162334], [839.8598859990575, 993.0162334], [839.8598859990575, 965.0162334], [839.0598859990575, 935.4162334], [836.8598859990575, 908.2162334], [823.8598859990575, 849.6162334], [885.0903845990574, 835.0080222], [923.3376637990575, 851.1121398]], [[782.393442442715, 946.2363379228119], [761.993442442715, 923.2363379228119], [743.5934424427151, 902.0363379228119], [715.1934424427151, 888.436337922812], [605.993442442715, 886.6363379228119], [606.7934424427151, 854.6363379228119], [724.393442442715, 851.2363379228119], [743.7934424427151, 831.4363379228118], [779.5934424427151, 821.6363379228119], [796.7934424427151, 855.4363379228118], [805.5934424427151, 874.6363379228119], [816.393442442715, 900.8363379228119], [825.641061442715, 925.1728459228119], [835.393442442715, 954.8363379228119], [798.498310442715, 964.2483613228119], [782.393442442715, 946.2363379228119]], [[53.0, 1022.9658114739605], [73.4, 1045.9658114739605], [91.80000000000001, 1067.1658114739605], [120.20000000000002, 1080.7658114739604], [229.40000000000003, 1082.5658114739606], [228.60000000000002, 1114.5658114739606], [111.00000000000001, 1117.9658114739605], [91.60000000000002, 1137.7658114739606], [55.80000000000001, 1147.5658114739606], [38.60000000000002, 1113.7658114739606], [29.80000000000001, 1094.5658114739606], [19.0, 1068.3658114739605], [9.752381000000014, 1044.0293034739605], [0.0, 1014.3658114739605], [36.89513200000002, 1004.9537880739605], [53.0, 1022.9658114739605]], [[53.0, 756.3027840346267], [73.4, 779.3027840346267], [91.80000000000001, 800.5027840346268], [120.20000000000002, 814.1027840346268], [229.40000000000003, 815.9027840346267], [228.60000000000002, 847.9027840346267], [111.00000000000001, 851.3027840346267], [91.60000000000002, 871.1027840346268], [55.80000000000001, 880.9027840346267], [38.60000000000002, 847.1027840346268], [29.80000000000001, 827.9027840346267], [19.0, 801.7027840346268], [9.752381000000014, 777.3662760346267], [0.0, 747.7027840346268], [36.89513200000002, 738.2907606346267], [53.0, 756.3027840346267]], [[973.6535461817058, 425.8731073784608], [994.0535461817058, 448.8731073784608], [1012.4535461817059, 470.0731073784608], [1040.8535461817057, 483.6731073784608], [1150.0535461817058, 485.47310737846084], [1149.2535461817058, 517.4731073784608], [1031.653546181706, 520.8731073784609], [1012.2535461817058, 540.6731073784608], [976.4535461817059, 550.4731073784608], [959.2535461817058, 516.6731073784608], [950.4535461817059, 497.47310737846084], [939.6535461817058, 471.27310737846085], [930.4059271817058, 446.9365993784608], [920.6535461817058, 417.27310737846085], [957.5486781817058, 407.8610839784608], [973.6535461817058, 425.8731073784608]], [[986.3047388030658, 132.740551292734], [965.9047388030658, 109.74055129273401], [947.5047388030658, 88.54055129273401], [919.1047388030659, 74.94055129273401], [809.9047388030658, 73.140551292734], [810.7047388030658, 41.140551292734], [928.3047388030658, 37.74055129273401], [947.7047388030659, 17.940551292734], [983.5047388030658, 8.140551292734017], [1000.7047388030659, 41.940551292734], [1009.5047388030658, 61.140551292734], [1020.3047388030658, 87.340551292734], [1029.552357803066, 111.67705929273401], [1039.304738803066, 141.340551292734], [1002.4096068030658, 150.75257469273402], [986.3047388030658, 132.740551292734]], [[53.0, 99.2097428954954], [73.4, 122.2097428954954], [91.80000000000001, 143.4097428954954], [120.20000000000002, 157.00974289549538], [229.40000000000003, 158.8097428954954], [228.60000000000002, 190.8097428954954], [111.00000000000001, 194.2097428954954], [91.60000000000002, 214.0097428954954], [55.80000000000001, 223.8097428954954], [38.60000000000002, 190.0097428954954], [29.80000000000001, 170.8097428954954], [19.0, 144.6097428954954], [9.752381000000014, 120.27323489549539], [0.0, 90.6097428954954], [36.89513200000002, 81.19771949549539], [53.0, 99.2097428954954]], [[282.7131193181219, 864.8453655604823], [262.27726851812196, 840.3223447604823], [261.71220071812195, 807.5484125604823], [270.4156957181219, 783.7588597604823], [287.02214311812196, 755.8114237604823], [303.31080491812196, 769.9282639604824], [321.5449747181219, 791.9350207604823], [333.51640331812195, 819.8683541604823], [331.62763771812195, 850.7181923604822], [311.31059191812193, 883.6124569604823], [282.7131193181219, 864.8453655604823]], [[571.971764972142, 618.4886064095963], [551.5359141721419, 593.9655856095964], [550.970846372142, 561.1916534095964], [559.674341372142, 537.4021006095963], [576.2807887721419, 509.4546646095963], [592.5694505721419, 523.5715048095963], [610.803620372142, 545.5782616095963], [622.775048972142, 573.5115950095964], [620.886283372142, 604.3614332095963], [600.569237572142, 637.2556978095963], [571.971764972142, 618.4886064095963]], [[406.1257986814308, 815.87921717601], [385.68994788143084, 791.3561963760101], [385.12488008143083, 758.5822641760101], [393.8283750814308, 734.7927113760101], [410.43482248143084, 706.84527537601], [426.72348428143084, 720.9621155760101], [444.9576540814308, 742.96887237601], [456.9290826814308, 770.9022057760101], [455.0403170814308, 801.75204397601], [434.7232712814308, 834.64630857601], [406.1257986814308, 815.87921717601]], [[667.0612724523774, 235.5927858232108], [646.6254216523773, 211.0697650232108], [646.0603538523774, 178.2958328232108], [654.7638488523774, 154.5062800232108], [671.3702962523773, 126.5588440232108], [687.6589580523773, 140.6756842232108], [705.8931278523773, 162.6824410232108], [717.8645564523774, 190.6157744232108], [715.9757908523774, 221.4656126232108], [695.6587450523774, 254.3598772232108], [667.0612724523774, 235.5927858232108]], [[786.8601130011033, 1098.6221532317838], [766.4242622011033, 1074.0991324317838], [765.8591944011033, 1041.3252002317838], [774.5626894011033, 1017.5356474317838], [791.1691368011033, 989.5882114317837], [807.4577986011033, 1003.7050516317838], [825.6919684011033, 1025.7118084317838], [837.6633970011034, 1053.6451418317838], [835.7746314011033, 1084.4949800317838], [815.4575856011033, 1117.3892446317839], [786.8601130011033, 1098.6221532317838]], [[738.8654750523773, 247.91272742321078], [718.4296242523773, 223.3897066232108], [717.8645564523773, 190.61577442321078], [726.5680514523773, 166.82622162321078], [743.1744988523773, 138.87878562321077], [759.4631606523773, 152.9956258232108], [777.6973304523773, 175.0023826232108], [789.6687590523774, 202.9357160232108], [787.7799934523773, 233.7855542232108], [767.4629476523774, 266.6798188232108], [738.8654750523773, 247.91272742321078]]]
    PltFunc.showPolys(polys)
