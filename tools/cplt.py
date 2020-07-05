from polygon import PltFunc
import pandas as pd
import json

def testCPlusResult():
    fu = pd.read_csv("/Users/sean/Documents/Projects/Packing-Algorithm/data/c_test.csv")
    _len = fu.shape[0]
    for i in range(_len):
        PltFunc.addPolygon(json.loads(fu["polygon"][i]))
    PltFunc.showPlt()