'''
2020年3月25日 Use Linear Programming Search New Position
'''
from tools.polygon import GeoFunc,PltFunc,getData,getConvex
from tools.packing import PolyListProcessor,NFPAssistant
from tools.heuristic import BottomLeftFill
from tools.lp import sovleLP
import pandas as pd
import json
from shapely.geometry import Polygon,Point,mapping,LineString
from interval import Interval
import copy
import random


class LPSearch(object):
    def __init__(self,width,original_polys):
        self.polys=copy.deepcopy(original_polys)
        self.NFPAssistant=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
        self.getInitialResult()
        
        self.divideNFP(self.NFPAssistant.getDirectNFP(self.polys[2],self.polys[10]))
        self.convertPolyToArea(self.NFPAssistant.getDirectNFP(self.polys[2],self.polys[10]))
        
    
    def getInitialResult(self):
        blf = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/blf.csv")
        self.polys=json.loads(blf["polys"][2])
        self.getLength()
    
    def shrink(self):
        self.cur_length=self.length*0.95
        
    def getLength(self):
        _max=0
        for i in range(0,len(self.polys)):
            extreme_index=GeoFunc.checkRight(self.polys[i])
            extreme=self.polys[i][extreme_index][1]
            if extreme>_max:
                _max=extreme
        self.length=_max
    
    def divideNFP(self,nfp):
        PltFunc.addPolygon(nfp)
        PltFunc.showPlt(width=2000,height=2000)
        print("获取nfp的拆分")
        # PltFunc.addP
    
    def getAngularBisector(self,pt1,pt2,pt3):
        '''
        输入：pt1/pt3为左右两个点，pt2为中间的点
        输出：对角线
        '''
        vec1=[pt1[0]-pt2[0],pt1[1]-pt2[1]]
        vec2=[pt3[0]-pt2[0],pt3[1]-pt2[1]]
    
    def convertPolyToArea(self,poly):
        '''
        将多边形区域转化为形状
        '''
        pass

    def getOverlapPair(self):
        '''
        获得两两形状间的重叠情况
        '''
        self.overlap_pair=[]
        for i in range(len(self.polys)-1):
            for j in range(i+1,len(self.polys)):
                pass

if __name__=='__main__':
    # polys=getConvex(num=5)
    polys=getData()
    # nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    # print(datetime.datetime.now(),"计算完成NFP")
    # blf=BottomLeftFill(1500,polys,vertical=True,NFPAssistant=nfp_ass)
    # print(blf.polygons)
    LPSearch(1500,polys)

