'''
本文件包括与DRL训练和测试的相关辅助函数
'''
import numpy as np
import multiprocessing
import time
from multiprocessing import Pool
from tqdm import tqdm
from heuristic import BottomLeftFill
from sequence import GA
from tools.packing import NFPAssistant,PolyListProcessor
from tools.polygon import getData
from train_data import GetBestSeq
max_point_num=4

def drop0(polys):
    '''
    网络输出的polys传入其他函数之前[必须完成]
    把所有多边形末尾的补零去掉
    '''
    polys_new=[]
    for poly in polys:
        for i in range(len(poly)):
            point_index=len(poly)-1-i
            if poly[point_index]==[0,0]:
                continue
            else:
                break
        poly=poly[0:point_index+1]
        polys_new.append(poly)
    return polys_new

def BLFwithSequence(test_path,width=800,seq_path=None,decrease=False,GA_algo=False):
    if seq_path!=None:
        f=open(seq_path,'r')
        seqs=f.readlines()
    data=np.load(test_path)
    size=data.shape[0]
    height=[]
    if GA_algo: p=Pool()
    multi_res=[]
    for i,line in enumerate(tqdm(data)):
        polys_new=[]
        polys_final=[]
        if seq_path!=None: # 指定序列
            seq=seqs[i].split(' ')
        else: # 随机序列
            seq=[0,1,2,3,4,5,6,7,8,9]
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
        polys_final=drop0(polys_final)
        if decrease==True: # 面积降序
            polys_final=GetBestSeq(width,polys_final).getDrease()            
        nfp_asst=NFPAssistant(polys_final,load_history=True,history_path='record/fu1500_val/{}.csv'.format(i))
        if GA_algo==True: # 遗传算法
            polys_GA=PolyListProcessor.getPolyObjectList(polys_final,[0])
            multi_res.append(p.apply_async(GA,args=(width,polys_GA,nfp_asst)))
        else:
            blf=BottomLeftFill(width,polys_final,NFPAssistant=nfp_asst)
            height.append(blf.getLength())
    if GA_algo:
        p.close()
        p.join()
        for i in range(size):
            height.append(multi_res[i].get().global_lowest_length)
    return height

def getBenchmark(source):
    random=BLFwithSequence(source)
    random=np.array(random)
    np.savetxt('random.CSV',random)
    print('random...OK')

    # predict=BLFwithSequence(source,seq_path='outputs/0404/fu1000/sequence-0.csv')
    # predict=np.array(predict)
    # np.savetxt('predict.CSV',predict)
    # print('predict...OK')

    decrease=BLFwithSequence(source,decrease=True)
    decrease=np.array(decrease)
    np.savetxt('decrease.CSV',decrease)
    print('decrease...OK')

    # ga=BLFwithSequence(source,decrease=True,GA_algo=True)
    # ga=np.array(ga)
    # np.savetxt('GA.CSV',ga)
    # print('GA...OK')

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
    center=[200,200] # 中心坐标
    for i in range(poly_num):
        point_num=np.random.randint(3,max_point_num+1)
        angle=360/point_num # 根据边数划分角度区域
        theta_start=np.random.randint(0,angle)
        for j in range(point_num):
            theta=(theta_start+angle*j)*np.pi/180 # 在每个区域中取角度并转为弧度
            #max_r=min(np.math.fabs(500/np.math.cos(theta)),np.math.fabs(500/np.math.sin(theta)))
            r=100+(160-100)*np.random.random()
            x=center[0]+r*np.math.cos(theta)
            y=center[1]+r*np.math.sin(theta)
            polys[i,2*j]=x
            polys[i,2*j+1]=y
            # print(theta,x,y)
    return polys
    
def generateTestData(size,poly_num=10,max_point_num=4):
    data=np.load('fu1500_val.npy')
    x=[]
    for i in range(size):
        polys=data[i]
        # polys=generatePolygon(poly_num,max_point_num)
        # polys=polys.T
        x.append(polys)
    x=np.array(x)
    np.save('test{}_{}_{}'.format(size,poly_num,max_point_num),x)

def getAllNFP(data_source,max_point_num):
    data=np.load(data_source)
    polys=[]
    for i in range(0,len(data)):
        line=data[i]
        poly_new=[]
        line=line.T
        for j in range(len(line)):
            poly_new.append(line[j].reshape(max_point_num,2).tolist())
        poly_new=drop0(poly_new)
        nfp_asst=NFPAssistant(poly_new,get_all_nfp=True,store_nfp=True,store_path='record/fu1500_val/{}.csv'.format(i))

def generateData_fu(poly_num):
    polys=np.zeros((poly_num,8)) # 最多4个点 x 2个坐标
    for i in range(poly_num):
        shape=np.random.randint(0,8) # 矩形 直角三角形 等腰三角形 直角梯形
        b=500
        a=25
        x=a+(b-a)*np.random.random()
        y=a+(b-a)*np.random.random()
        if shape==0 or shape==1:
            points=[0,0,x,0,x,y,0,y]
        elif shape==2:
            points=[0,0,x,0,x,y,0,0]
        elif shape==3:
            points=[0,0,x,y,0,y,0,0]
        elif shape==4:
            points=[0,0,x,0,x/2,y,0,0]
        elif shape==5:
            points=[0,0,x,y/2,0,y,0,0]
        elif shape==6:
            x2=a+(b-a)*np.random.random()
            points=[0,0,x2,0,x,y,0,y]
        elif shape==7:
            y2=a+(b-a)*np.random.random()
            points=[0,0,x,0,x,y2,0,y]
        polys[i]=points
    return polys # [ poly_num x (max_point_num * 2) ]  

def polys2data(polys):
    '''
    将poly进行reshape满足网络输入格式
    '''
    max_point_num=0
    size=len(polys)
    for poly in polys:
        point_num=len(poly)
        if point_num>max_point_num:
            max_point_num=point_num
    polys_new=np.zeros((size,max_point_num*2))
    for i in range(size):
        poly=polys[i]
        point_num=len(poly)
        poly=np.array(poly)
        poly=poly.reshape(1,point_num*2)
        poly=poly[0]
        for index,point in enumerate(poly):
            polys_new[i][index]=point
    return polys_new


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn',True) 
    start=time.time()
    # getAllNFP('fu1500_val.npy',4)
    #generateTestData(900)
    # data=np.load('fu.npy',allow_pickle=True)
    # print(data.shape)
    getBenchmark('test900.npy')
    end=time.time()
    print(end-start)
