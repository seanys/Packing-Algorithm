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
        all_new_seq=self.getAllSeq(_list)
    
    def checkOneSeq(self):
        pass
        # 获得排样情况
        # new_polys=[]
        # for item in new_list:
        #     new_polys.append(self.polys[item[0]])

        # nfp_assistant=NFPAssistant(new_polys,store_nfp=False,get_all_nfp=True,load_history=True)
        # packing_polys=BottomLeftFill(760,new_polys,NFPAssistant=nfp_assistant).polygons
        # _len=LPAssistant.getLength(packing_polys)

        # # print("利用率：",433200/(_len*760))
        # res=[[] for i in range(len(new_polys))]
        # for i,item in enumerate(new_list):
        #     res[new_list[i][0]]=packing_polys[i]

        # return res

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
        print(new_list)
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
        


class Clustering(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    initialResult(getData())
