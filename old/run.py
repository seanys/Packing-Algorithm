import pandas as pd
import csv
import numpy as np
from tools.vectorization import vectorFunc
from tools.nfp import NFP,PlacePolygons
from tools.dp import getDP
from tools.geom import graphCV
from tools.fittest import fittestPosition
from shapely.geometry import Polygon,Point,mapping,LineString
import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(filename)s-line%(lineno)d:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")


class dataEvaluate(object):
    '''
    图形量化并rebuild的效果测试
    '''
    def __init__(self):
        self.data=[]
        self.file_name="50_han"
        self.data=pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/euro_data/standard/"+self.file_name+".csv")
        self.evaluateData()

    def evaluateData(self):
        _arr=[[],[],[],[]]
        for index, row in self.data.iterrows():
            logging.log(logging.INFO,"————————————第",index,"轮——————————————")

            vertexes=json.loads(row['polygon']) # 加载数据
            self.normData(vertexes) # 位置标准化
            qf=quantiFunc(vertexes) # 量化数据
            qf.quantification() # 量化数据
            rev=rebuildEvalute(qf.vector,qf) 
            _arr[0].append(rev.contain_ratio)
            _arr[1].append(rev.exceed_ratio)
            _arr[2].append(rev.inter_area)
            _arr[3].append(qf.centroid_in)
        
        logging.log(logging.INFO,_arr)
        # 输出最终计算情况
        # with open("/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/rebuild_res/"+self.file_name+"_res.csv","a+") as csvfile:
        #     writer = csv.writer(csvfile)
        #     _len=len(inter_area)
        #     for i in range(_len):
        #         writer.writerows([[contain_ratio[i],exceed_ratio[i],inter_area[i],centroid_in[i]]])
    
    def normData(self,vertexes):
        '''
        图形位置标准化，图形向上向左移动
        '''
        P=Polygon(vertexes)
        delta_x=P.bounds[0]-100
        delta_y=P.bounds[1]-100
        for ver in vertexes:
            ver[0]=round(ver[0]-delta_x,4)
            ver[1]=round(ver[1]-delta_y,4)

class combinationTest(object):
    def testNFP():
        triangle=[[10, 17.0], [72, 17.0], [10, 65.0]]
        rectangle=[[17.0, 10], [30.0, 20], [100.0, 10], [120.0, 60], [51.0, 80], [30.0, 50], [17.0, 58],[30.0, 30]]
        poly1=rectangle
        poly2=triangle
        combinationTest.dataNorm(poly1,3)
        combinationTest.dataNorm(poly2,3)
        
        # NFP计算 多边形，多边形，是否显示最终结果
        nfp=NFP(poly1,poly2,False)
        # Most Fit Position计算
        mfp=MostFitPosition(nfp,True)
    
    def dataNorm(poly,multi):
        for pt in poly:
            pt[0]=pt[0]*multi
            pt[1]=pt[1]*multi

class combinationInput(object):
    def testNFP(polygons):
        # 数据标准化
        for poly in polygons:
            combinationTest.dataNorm(poly,1)
        
        # NFP计算 多边形，多边形，是否显示最终结果
        nfp=NFP(poly1,poly2,False)
        # Most Fit Position计算
        mfp=MostFitPosition(nfp,True)
    
    def dataNorm(poly,multi):
        for pt in poly:
            pt[0]=pt[0]*multi
            pt[1]=pt[1]*multi

if __name__=='__main__':
    # de=dataEvaluate() # 测试量化结果
    # combinationTest.testNFP() # 测试NFP排样
    gCV=graphCV()
    poly=[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [4.0, 5.0], [3.0, 3.0], [2.0, 2.0], [0.0, 1.0]]
    combinationInput.dataNorm(poly,100)
    gCV.addPolygonColor(poly)
    gCV.showPolygon()