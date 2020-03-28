import pandas as pd
import json
import numpy as np
import os
def getTrainData(point_num):
    """
    point_num:取样点个数
    return:x,y,x_test,y_test
    """
    if os.getlogin()=='Prinway':
        path=r'D:\\Tongji\\Nesting\\Data\\'
    else:
        path='/Users/sean/Documents/Projects/Data/'
    X,Y,x_test,y_test=[],[],[],[]
    _input=pd.read_csv(path+'input_seq.csv')
    _output=pd.read_csv(path+'output_order.csv')
    # 训练集
    for i in range(0,4000):
        X.append(json.loads(_input["x_"+str(point_num)][i]))
        Y.append(json.loads(_output["y_order"][i]))
    # 测试集
    for i in range(4000,5000):
        x_test.append(json.loads(_input["x_"+str(point_num)][i]))
        y_test.append(json.loads(_output["y_order"][i]))

    return np.array(X),np.array(Y),np.array(x_test),np.array(y_test)
