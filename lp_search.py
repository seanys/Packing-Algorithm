'''
2020年3月25日 Use Linear Programming Search New Position
'''
from tools.polygon import GeoFunc,PltFunc,getData,getConvex
from sequence import BottomLeftFill,PolyListProcessor,NFPAssistant
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
    
    def convertPolyToArea(self,poly):
        pass
    

if __name__=='__main__':
    # polys=getConvex(num=5)
    polys=getData()
    # nfp_ass=NFPAssistant(polys,store_nfp=False,get_all_nfp=True,load_history=True)
    # print(datetime.datetime.now(),"计算完成NFP")
    # blf=BottomLeftFill(1500,polys,vertical=True,NFPAssistant=nfp_ass)
    # print(blf.polygons)
    LPSearch(1500,polys)

