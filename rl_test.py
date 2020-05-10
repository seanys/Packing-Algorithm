'''
本文件包括与DRL训练和测试的相关辅助函数
'''
import numpy as np
import pandas as pd
import time
import multiprocessing
import os
import json
import itertools
from shutil import copyfile
from multiprocessing import Pool
from tqdm import tqdm
from heuristic import BottomLeftFill,RatotionPoly
from sequence import GA
from shapely.geometry import Polygon
from tools.packing import NFPAssistant,PolyListProcessor
from tools.polygon import getData,GeoFunc,PltFunc
from tools.vectorization import vectorFunc

class GenerateData_xy(object):
    '''
    04/09后采用vector方法生成 弃用此类
    '''

    @staticmethod
    def generateRectangle(poly_num,max_width,max_height):
        polys=np.zeros((poly_num,8)) # 4个点 x 2个坐标
        for i in range(poly_num):
            x=np.random.randint(50,max_width)
            y=np.random.randint(50,max_height)
            points=[0,0,x,0,x,y,0,y]
            polys[i]=points
        return polys

    @staticmethod
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
        
    @staticmethod
    def generateTestData(size,poly_num=10,max_point_num=4):
        x=[]
        for i in range(size):
            polys=polys2data(getData(index=5))
            # polys=generatePolygon(poly_num,max_point_num)
            polys=polys.T
            x.append(polys)
        x=np.array(x)
        np.save('test{}_{}_{}'.format(size,poly_num,max_point_num),x)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

class GenerateData_vector(object):

    @staticmethod
    def generateSpecialPolygon(shape):
        # shape: 1 直角三角形 2 等腰三角形 3 矩形 4 直角梯形 5 菱形
        b=250
        a=100
        x=a+(b-a)*np.random.random()
        y=a+(b-a)*np.random.random()
        if shape==1:
            poly=np.array([0,0,x,0,0,y]).reshape(3,2).tolist()
            RatotionPoly(90).rotation(poly)
        elif shape==2:
            poly=np.array([0,0,x,0,x/2,y]).reshape(3,2).tolist()
            RatotionPoly(90).rotation(poly)
        elif shape==3:
            poly=np.array([0,0,x,0,x,y,0,y]).reshape(4,2).tolist()
            RatotionPoly(90).rotation(poly)
        elif shape==4:
            x2=a+(b-a)*np.random.random()
            poly=np.array([0,0,x2,0,x,y,0,y]).reshape(4,2).tolist()
            RatotionPoly(90).rotation(poly)
        elif shape==5:
            poly=np.array([0,0,x/2,-y/2,x,0,x/2,y/2]).reshape(4,2).tolist()
            RatotionPoly(90).rotation(poly)
        return poly
    
    @staticmethod
    def generatePolygon(point_num,is_regular):
        '''
        随机生成多边形
        point_num: 点的个数
        is_regular: 是否正多边形
        '''
        r_max=140
        r_min=60
        poly=[]
        angle=360/point_num # 根据边数划分角度区域
        r=r_min+(r_max-r_min)*np.random.random()
        for j in range(point_num):
            theta_min=angle*j
            theta_max=angle*(j+1)
            theta=theta_min if is_regular else theta_min+(theta_max-theta_min)*np.random.random()         
            theta=theta*np.pi/180 # 角度转弧度
            #max_r=min(np.math.fabs(500/np.math.cos(theta)),np.math.fabs(500/np.math.sin(theta)))
            if not is_regular:
                r=r_min+(r_max-r_min)*np.random.random()
            x=r*np.math.cos(theta)
            y=r*np.math.sin(theta)
            poly.append([x,y])
        if is_regular:
            if point_num==5 or point_num==7:
                RatotionPoly(360).rotation_specific(poly,angle=[i*90 for i in range(4)])
            elif point_num==6:
                RatotionPoly(360).rotation_specific(poly,angle=[0,90])
            elif point_num==8:
                RatotionPoly(360).rotation_specific(poly,angle=[0,45/2])
        return poly
    
    @staticmethod
    def generateTestData(dataset_name,size):
        data=[]
        vectors=[]
        for i in tqdm(range(size)):
            polys=[]
            for j in range(12):
                polyCheck=False
                while not polyCheck:
                    dice=np.random.random()
                    if dice<1:
                        shape=np.random.randint(1,6)
                        poly=GenerateData_vector.generateSpecialPolygon(shape)
                    elif dice<0.7:
                        point_num=np.random.randint(3,9)
                        poly=GenerateData_vector.generatePolygon(point_num,True)
                    else:
                        point_num=np.random.randint(3,9)
                        poly=GenerateData_vector.generatePolygon(point_num,False)
                    if Polygon(poly).area>5000: # 面积过小会导致无法计算NFP
                        polyCheck=True
                polys.append(poly)
            # blf=BottomLeftFill(760,polys)
            # blf.showAll()
            data.append(polys)
            vector=[]
            for poly in polys:
                vector.append(vectorFunc(poly,cut_nums=128).vector)
            vectors.append(vector)
        data=np.array(data)
        vectors=np.array(vectors)
        np.save('{}_xy'.format(dataset_name),data)
        np.save('{}'.format(dataset_name),vectors)

    @staticmethod
    def poly2vector(source,save_name):
        data=np.load(source,allow_pickle=True)
        vectors=[]
        for index,line in enumerate(tqdm(data)):
            vector=[]
            for poly in line:
                vector.append(vectorFunc(poly,cut_nums=128).vector)
            vectors.append(vector)
        vectors=np.array(vectors)
        np.save(save_name,vectors)

    @staticmethod
    def exportDataset(index,export_name):
        data=[]
        vectors=[]
        polys=getData(index)
        data.append(polys)
        vector=[]
        for poly in polys:
            vector.append(vectorFunc(poly,cut_nums=128).vector)
        vectors.append(vector)
        data=np.array(data)
        vectors=np.array(vectors)
        np.save('{}_xy'.format(export_name),data)
        np.save('{}'.format(export_name),vectors)

class InitSeq(object):
    def __init__(self,width,polys,nfp_load=None):
        self.polys=polys
        self.width=width
        if nfp_load!=None:
            self.NFPAssistant=NFPAssistant(polys,load_history=True,history_path=nfp_load)
        else:
            self.NFPAssistant=None
        
    # 获得面积/长度/高度降序排列的形状结果
    def getDrease(self,criteria):
        poly_list=[]
        for poly in self.polys:
            if criteria=='length':
                left,bottom,right,top=GeoFunc.checkBoundValue(poly)          
                poly_list.append([poly,right-left])
            elif criteria=='height':
                left,bottom,right,top=GeoFunc.checkBoundValue(poly)    
                poly_list.append([poly,top-bottom])
            else:
                poly_list.append([poly,Polygon(poly).area])
        poly_list=sorted(poly_list, key = lambda item:item[1], reverse = True) # 排序，包含index
        dec_polys=[]
        for item in poly_list:
            dec_polys.append(item)
        return dec_polys

    # 获得所有降序排列方案的最优解
    def getBest(self):
        min_height=999999999
        heights=[]
        best_criteria=''
        for criteria in ['area','length','height']:
            init_list=self.getDrease(criteria)
            # 获得全部聚类结果
            clustering,now_clustering,last_value=[],[],init_list[0][1]
            for item in init_list:
                if item[1]==last_value:
                    now_clustering.append(item)
                else:
                    clustering.append(now_clustering)
                    last_value=item[1]
                    now_clustering=[item]
            clustering.append(now_clustering)
            # 获得全部序列
            one_lists=[]
            for item in clustering:
                one_list=list(itertools.permutations(item))
                one_lists.append(one_list)
            all_lists=itertools.product(*one_lists)
            lists=[]
            for cur_lists in all_lists:
                cur_list=[]
                for polys in cur_lists:
                    for poly in polys:
                        cur_list.append(poly)
                lists.append(cur_list)
            for item in lists:
                polys_final=[]
                for poly in item:
                    polys_final.append(poly[0])
                blf=BottomLeftFill(self.width,polys_final,NFPAssistant=self.NFPAssistant)
                height=blf.getLength()
                heights.append(height)
                if height<min_height:
                    min_height=height
                    best_criteria=criteria
        # print(sorted(heights,reverse=False))
        # print(min_height,best_criteria)
        area=0
        for poly in self.polys:
            area=area+Polygon(poly).area
        use_ratio=area/(self.width*min_height)
        # print(area,use_ratio)
        return min_height


    # 枚举所有序列并选择最优
    def getAll(self):
        all_com=list(itertools.permutations([(i) for i in range(len(self.polys))]))
        min_height=999999999
        best_order=[]
        for item in all_com:
            seq=self.getPolys(item)
            height=BottomLeftFill(self.width,seq,NFPAssistant=self.NFPAssistant).getLength()
            if height<min_height:
                best_order=item
                min_height=height
        area=0
        for poly in self.polys:
            area=area+Polygon(poly).area
        use_ratio=area/(self.width*min_height)
        return best_order,min_height,use_ratio
    
    def getPolys(self,seq):
        seq_polys=[]
        for i in seq:
            seq_polys.append(self.polys[i])
        return seq_polys

def getNFP(polys,save_name,index):
    # print('record/{}/{}.csv'.format(save_name,index))
    NFPAssistant(polys,get_all_nfp=True,store_nfp=True,store_path='record/{}/{}.csv'.format(save_name,index))

def getAllNFP(data_source,save_name):
    os.makedirs('record/{}'.format(save_name))
    data=np.load(data_source,allow_pickle=True)
    p=Pool()
    for index,polys in enumerate(data):
        p.apply_async(getNFP,args=(polys,save_name,index))
    p.close()
    p.join()

def NFPcheck(dataset_name,new_name):
    os.makedirs('record/{}'.format(new_name))
    print('Files with wrong NFPs are listed below:')
    xy=np.load('{}_xy.npy'.format(dataset_name),allow_pickle=True)
    vec=np.load('{}.npy'.format(dataset_name),allow_pickle=True)
    xy_new=[]
    vec_new=[]
    index_new=0
    for i,polys in enumerate(tqdm(xy)):
        valid=False
        nfp_path='record/{}/{}.csv'.format(dataset_name,i)
        if not os.path.exists(nfp_path):
            continue
        df = pd.read_csv(nfp_path,header=None)
        try:
            for line in range(df.shape[0]):
                nfp=json.loads(df[2][line])
                differ=Polygon([[-1000,-1000],[3000,-1000],[3000,3000],[-1000,3000]]).difference(Polygon(nfp))
            valid=True
        except:
            print(i)
        if valid:
            xy_new.append(xy[i])
            vec_new.append(vec[i])
            copyfile('record/{}/{}.csv'.format(dataset_name,i),'record/{}/{}.csv'.format(new_name,index_new))
            index_new=index_new+1
    print('数据集有效容量 {}'.format(len(vec_new)))
    np.save('{}_xy.npy'.format(new_name),np.array(xy_new))
    np.save('{}.npy'.format(new_name),np.array(vec_new))

def BLFwithSequence(test_path,width,seq_path=None,GA_algo=False):
    if seq_path!=None:
        f=open(seq_path,'r')
        seqs=f.readlines()
    data=np.load(test_path,allow_pickle=True)
    test_name=test_path.split('_xy')[0]
    height=[]
    if GA_algo: p=Pool()
    multi_res=[]
    for i,line in enumerate(tqdm(data)):
        polys_final=[]
        if seq_path!=None: # 指定序列
            seq=seqs[i].split(' ')
        else: # 随机序列
            seq=np.array(range(len(line)))
            np.random.shuffle(seq)
        for j in range(len(line)):
            if seq_path!=None:
                index=int(seq[j])
            else:
                index=seq[j]
            polys_final.append(line[index])           
        nfp_asst=NFPAssistant(polys_final,load_history=True,history_path='record/{}/{}.csv'.format(test_name,i))
        #nfp_asst=None
        if GA_algo==True: # 遗传算法
            polys_GA=PolyListProcessor.getPolyObjectList(polys_final,[0])
            multi_res.append(p.apply_async(GA,args=(width,polys_GA,nfp_asst)))
        else:
            blf=BottomLeftFill(width,polys_final,NFPAssistant=nfp_asst)
            #blf.showAll()
            height.append(blf.getLength())
    if GA_algo:
        p.close()
        p.join()
        for res in multi_res:
            height.append(res.get().global_lowest_length)
    return height

def getBenchmark(source,width=760):
    random=BLFwithSequence(source,width)
    random=np.array(random)
    # np.savetxt('random.CSV',random)
    print('random...OK',np.mean(random))

    # 与现有启发式比较
    data=np.load(source,allow_pickle=True)
    test_name=source.split('_xy')[0]
    height=[]
    for i,line in enumerate(tqdm(data)):
        nfp_path='record/{}/{}.csv'.format(test_name,i)
        min_height=InitSeq(width,line,nfp_load=nfp_path).getBest()
        height.append(min_height)
    decrease=np.array(height)
    print('heuristic...OK',np.mean(decrease))

    # predict=BLFwithSequence(source,seq_path='outputs/0406/fu1500/sequence-0.csv')
    # predict=np.array(predict)
    # np.savetxt('predict.CSV',predict)
    # print('predict...OK')

    # ga=BLFwithSequence(source,decrease=None,GA_algo=True)
    # if single:  print('GA',ga)
    # else:
    #     ga=np.array(ga)
    #     np.savetxt('GA.CSV',ga)
    #     print('GA...OK')

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn',True) 
    start=time.time()
    # GenerateData_vector.exportDataset(4,'dighe1')
    # GenerateData_vector.exportDataset(5,'dighe2')
    # NFPcheck('fu999_val','reg9999_val')
    # GenerateData_vector.generateTestData('fu999_val',999)
    # getAllNFP('fu999_val_xy.npy','fu999_val')
    # GenerateData_vector.generateTestData('reg10000',10000)
    # getAllNFP('reg10000_xy.npy','reg10000')
    getBenchmark('reg2379_xy.npy')
    # data=np.load('fu_val_xy.npy',allow_pickle=True)[0]
    # InitSeq(760,data,nfp_load='record/fu_val/0.csv').getBest()
    #InitSeq(760,data,nfp_load='record/fu_10_val/0.csv').getBest()
    end=time.time()
    print('Running time:',end-start)
