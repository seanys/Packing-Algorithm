from tools.polygon import PltFunc,GeoFunc,NFP,getData
from sequence import BottomLeftFill
from tools.packing import NFPAssistant
from tools.lp_assistant import LPAssistant
from shapely.geometry import Polygon,mapping
from shapely import affinity
from lp_search import LPSearch
import pandas as pd # 读csv
import csv # 写csv
import json
import itertools

class PreProccess(object):
    '''
    预处理NFP以及NFP divided函数
    '''
    def __init__(self):
        self.main()
        # self.orientation()

    def orientation(self):
        fu = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/data/fu.csv")
        _len= fu.shape[0]
        min_angle=90
        rotation_range=[0,1,2,3]
        with open("/Users/sean/Documents/Projects/Data/fu_orientation.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(_len):
                Poly_i=Polygon(self.normData(json.loads(fu["polygon"][i])))
                all_poly=[]
                for oi in rotation_range:
                    all_poly.append(self.rotation(Poly_i,oi,min_angle))
                writer.writerows([all_poly])


    def main(self):
        fu = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/data/fu.csv")
        _len= fu.shape[0]
        rotation_range=[0,1,2,3]
        min_angle=90
        with open("/Users/sean/Documents/Projects/Data/fu.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(_len):
                Poly_i=Polygon(self.normData(json.loads(fu["polygon"][i])))
                for j in range(_len):
                    Poly_j=Polygon(self.normData(json.loads(fu["polygon"][j])))
                    for oi in rotation_range:
                        new_poly_i=self.rotation(Poly_i,oi,min_angle)
                        self.slideToOrigin(new_poly_i)
                        for oj in rotation_range:
                            print(i,j,oi,oj)
                            new_poly_j=self.rotation(Poly_j,oj,min_angle)
                            nfp=NFP(new_poly_i,new_poly_j)
                            new_nfp=LPSearch.deleteOnline(nfp.nfp)
                            all_bisectior,divided_nfp,target_func=LPSearch.getDividedNfp(new_nfp)
                            writer.writerows([[i,j,oi,oj,new_poly_i,new_poly_j,new_nfp,divided_nfp,target_func]])

    def slideToOrigin(self,poly):
        bottom_pt,min_y=[],999999999
        for pt in poly:
            if pt[1]<min_y:
                min_y=pt[1]
                bottom_pt=[pt[0],pt[1]]
        GeoFunc.slidePoly(poly,-bottom_pt[0],-bottom_pt[1])

    def normData(self,poly):
        new_poly,num=[],20
        for pt in poly:
            new_poly.append([pt[0]*num,pt[1]*num])
        return new_poly

    def rotation(self,Poly,orientation,min_angle):
        if orientation==0:
            return self.getPoint(Poly)
        new_Poly=affinity.rotate(Poly,min_angle*orientation)
        return self.getPoint(new_Poly)
    
    def getPoint(self,shapely_object):
        mapping_res=mapping(shapely_object)
        coordinates=mapping_res["coordinates"][0]
        new_poly=[]
        for pt in coordinates:
            new_poly.append([pt[0],pt[1]])
        return new_poly


class initialResult(object):
    def __init__(self,polys):
        self.polys=polys
        self.main(_type="width")
    
    def main(self,_type):
        _list=[]
        if _type=="area":
            pass
        elif _type=="length":
            pass
        elif _type=="width":
            _list=self.getWidthDecreaing()
        elif _type=="rectangularity":
            pass
        else:
            pass
        # 重排列后的结果
        self.nfp_assistant=NFPAssistant(self.polys,store_nfp=False,get_all_nfp=True,load_history=True)
        PltFunc.showPolys([[[496.0, 64.0], [696.0, 64.0], [696.0, 264.0], [496.0, 264.0]], [[490.0, 500.0], [690.0, 500.0], [690.0, 700.0], [490.0, 700.0]], [[210.0, 580.0], [490.0, 580.0], [490.0, 760.0], [210.0, 760.0]], [[280.0, 0.0], [560.0, 0.0], [420.0, 140.0]], [[0.0, 460.0], [0.0, 280.0], [280.0, 460.0]], [[0.0, 0.0], [280.0, 0.0], [280.0, 280.0], [0.0, 280.0]], [[280.0, 84.0], [480.0, 164.0], [480.0, 264.0], [280.0, 264.0]], [[580.0, 264.0], [680.0, 264.0], [680.0, 444.0], [580.0, 444.0]], [[100.0, 280.0], [380.0, 280.0], [380.0, 560.0]], [[380.0, 264.0], [580.0, 264.0], [580.0, 464.0], [380.0, 544.0]], [[0.0, 757.1428571428571], [80.0, 597.1428571428571], [160.0, 757.1428571428571]], [[0.0, 460.0], [280.0, 460.0], [140.0, 700.0]]])
        all_list = pd.read_csv("/Users/sean/Documents/Projects/Data/all_list.csv")
        for i in range(10000,11000):
            seq=json.loads(all_list["list"][i])
            ratio,result=self.checkOneSeq(seq)
            if ratio>0.77:
                print(i,ratio,result)
    
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

    def getLengthDecreaing(self,polys):

        pass

    def getWidthDecreaing(self):
        width_list=[]
        for i,poly in enumerate(self.polys):
            left_pt,right_pt=LPAssistant.getLeftPoint(poly),LPAssistant.getRightPoint(poly)
            width_list.append([i,right_pt[0]-left_pt[0]])
        return width_list

    def getRectangularityDecreaing(self,polys):
        
        pass

    def getAllSeq(self,_list):
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

if __name__ == '__main__':
    initialResult(getData())
