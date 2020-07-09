from polygon import PltFunc
import pandas as pd
import json

def testCPlusResult():
    fu = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/c_plt/c_record.csv")
    _len = fu.shape[0]
    for i in range(_len):
        PltFunc.addPolygon(json.loads(fu["polygon"][i]))

    container = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/record/c_plt/container.csv")
    width = container["container"][0]
    length = container["container"][1]
    PltFunc.addPolygonColor([[0,0],[length,0],[length,width],[0,width]])

    PltFunc.showPlt()

if __name__ == '__main__':
    testCPlusResult()