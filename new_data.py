from tools.polygon import PltFunc,GeoFunc,NFP
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
        self.simplify()
        # self.main()
        # self.orientation()

    def simplify(self):        
        fu = pd.read_csv("/Users/sean/Documents/Projects/Data/fu.csv")
        with open("/Users/sean/Documents/Projects/Data/fu_simplify.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(fu.shape[0]):
                # i,j,oi,oj,new_poly_i,new_poly_j,nfp
                writer.writerows([[fu["i"][i],fu["j"][i],fu["oi"][i],fu["oj"][i],fu["new_poly_i"][i],fu["new_poly_j"][i],fu["nfp"][i]]])
                

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
    def getAreaDecreaing(self,polys):

        pass

    def getLengthDecreaing(self,polys):
    
        pass

    def getWidthDecreaing(self,polys):
        
        pass

    def getRectangularityDecreaing(self,polys):
        
        pass


if __name__ == '__main__':
    PreProccess()
