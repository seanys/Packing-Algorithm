'''
将预测结果可行化
1. 直接按照重心的高低排列，逐一添加
2. 通过比较高的
'''
from polygon import GeoFunc
from sequence import BottomLeftFill

# Guided Local Search的变体
class variantGLS(object):
    def __init__(self,polys):
        self.polys=polys


# 通过预测出的高度形成序列再采用Bottom Left Fill实现
def getFeasibleByBottom(polys):
    polyList=[]
    for poly in polys:
        bottom=poly[GeoFunc.checkBottom(poly)]
        polyList.append({
            "poly":poly,
            "bottom_x":bottom[0],
            "bottom_y":bottom[1]
        })
    sorted(polyList,key=lambda poly: poly["bottom_y"])
    sequence=[]
    for item in polyList:
        sequence.append(item["poly"])
    pp=BottomLeftFill(1500,sequence)
    return pp.polygons
