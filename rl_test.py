'''
本文件包括与DRL训练和测试的相关辅助函数
'''
import numpy as np
from tools.heuristic import BottomLeftFill
from tools.packing import NFPAssistant
max_point_num=5

def BLFwithSequence(test_path,seq_path=None):
    if seq_path!=None:
        f=open(seq_path,'r')
        seqs=f.readlines()
    data=np.load(test_path)
    size=data.shape[0]
    height=[]
    for i in range(size):
        polys_new=[]
        polys_final=[]
        line=data[i]
        if seq_path!=None:
            seq=seqs[i].split(' ')
        else:
            seq=[0,1,2,3,4,5,6,7]
            np.random.shuffle(seq)
        line=line.T
        for polys in line:
            poly=polys.reshape(max_point_num,2).tolist()
            polys_new.append(poly)
        for j in range(len(polys_new)):
            if seq_path!=None:
                index=int(seq[j])
            else:
                index=seq[j]
            polys_final.append(polys_new[index])
        blf=BottomLeftFill(1000,polys_final,vertical=True)
        height.append(blf.getLength())
    return height

def getBenchmark():
    random=BLFwithSequence(r'D:\\Tongji\\Nesting\\Data\\test200_8_5.npy')
    random=np.array(random)
    np.savetxt('randomSEQ.CSV',random)

    predict=BLFwithSequence(r'D:\\Tongji\\Nesting\\Data\\test200_8_5.npy','outputs/seq2000/032823/sequence-3.csv')
    predict=np.array(predict)
    np.savetxt('predictSEQ.CSV',predict)

def generateRectangle(poly_num,max_width,max_height):
    polys=np.zeros((poly_num,8)) # 4个点 x 2个坐标
    for i in range(poly_num):
        x=np.random.randint(50,max_width)
        y=np.random.randint(50,max_height)
        points=[0,0,x,0,x,y,0,y]
        polys[i]=points
    return polys

def generatePolygon(poly_num,max_point_num):
    '''
    随机生成多边形
    poly_num: 多边形个数
    max_point_num: 最大点的个数
    '''
    polys=np.zeros((poly_num,max_point_num*2))
    center=[250,250] # 中心坐标
    for i in range(poly_num):
        point_num=np.random.randint(5,max_point_num+1)
        angle=360/point_num # 根据边数划分角度区域
        for j in range(point_num):
            theta=np.random.randint(angle*j,angle*(j+1))*np.pi/180 # 在每个区域中取随机角度并转为弧度
            #max_r=min(np.math.fabs(500/np.math.cos(theta)),np.math.fabs(500/np.math.sin(theta)))
            #r=np.random.randint(0,max_r) # 取随机长度
            r=np.random.randint(25,250) # 降低难度
            x=center[0]+r*np.math.cos(theta)
            y=center[1]+r*np.math.sin(theta)
            polys[i,2*j]=x
            polys[i,2*j+1]=y
            # print(theta,x,y)
    return polys
    
def generateTestData(size,poly_num=10,max_point_num=5,rectangle=False):
    x=[]
    for i in range(size):
        if not rectangle:
            polys=generatePolygon(poly_num,max_point_num)
        else:
            polys=generateRectangle(poly_num,500,500)
        polys=polys.T
        x.append(polys)
    x=np.array(x)
    np.save('test{}_{}_{}'.format(size,poly_num,max_point_num),x)

def chooseRectangle(data_source,size,is_train=True):
    data=np.loadtxt(data_source)
    x=[]
    for i in range(size):
        if is_train:
            index=np.random.choice(list(range(50)),10)
        else:
            index=np.random.choice(list(range(50,100)),10)
        polys=[]
        for j in range(len(index)):
            polys.append(data[index[j]].tolist())
        polys=np.array(polys)
        polys=polys.T
        x.append(polys)
    x=np.array(x)
    np.save('rec500_val',x)

def getAllNFP(data_source,max_point_num):
    data=np.loadtxt(data_source)
    polys=[]
    for line in data:
        poly=line.reshape(max_point_num,2).tolist()
        polys.append(poly)
    nfp_asst=NFPAssistant(polys,get_all_nfp=True,store_nfp=True,store_path='record/rec100_nfp.csv')

if __name__ == "__main__":
    #np.savetxt('data/rec100.csv',generateRectangle(100,500,500),fmt='%.2f')
    #getAllNFP('data/rec100.csv',4)
    generateTestData(10000,rectangle=True)
    pass