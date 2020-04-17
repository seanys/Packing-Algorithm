'''
本文件包括与DRL训练和测试的相关辅助函数
'''
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from heuristic import BottomLeftFill
from sequence import GA
from shapely.geometry import Polygon
from tools.packing import NFPAssistant,PolyListProcessor
from tools.polygon import getData,GeoFunc
from tools.vectorization import vectorFunc


def getNFP(polys,save_name,index):
    # print('record/{}/{}.csv'.format(save_name,index))
    NFPAssistant(polys,get_all_nfp=True,store_nfp=True,store_path='record/{}/{}.csv'.format(save_name,index))

def getAllNFP(data_source,save_name):
    data=np.load(data_source,allow_pickle=True)
    p=Pool()
    for index,polys in enumerate(data):
        p.apply_async(getNFP,args=(polys,save_name,index))
    p.close()
    p.join()

def BLFwithSequence(test_path,width=800,seq_path=None,decrease=None,GA_algo=False):
    if seq_path!=None:
        f=open(seq_path,'r')
        seqs=f.readlines()
    data=np.load(test_path,allow_pickle=True)
    size=data.shape[0]
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
        if decrease!=None: # 降序
            polys_final=GetBestSeq(width,polys_final,criteria=decrease).getDrease()            
        #nfp_asst=NFPAssistant(polys_final,load_history=True,history_path='record/fu_10_val/{}.csv'.format(i))
        nfp_asst=None
        if GA_algo==True: # 遗传算法
            polys_GA=PolyListProcessor.getPolyObjectList(polys_final,[0])
            multi_res.append(p.apply_async(GA,args=(width,polys_GA,nfp_asst,50,10)))
        else:
            blf=BottomLeftFill(width,polys_final,NFPAssistant=nfp_asst)
            #blf.showAll()
            height.append(blf.getLength())
    if GA_algo:
        p.close()
        p.join()
        for i in range(size):
            height.append(multi_res[i].get().global_lowest_length)
    return height

def getBenchmark(source,single=False):
    random=BLFwithSequence(source)
    if single:  print('random',random)
    else:
        random=np.array(random)
        np.savetxt('random.CSV',random)
        print('random...OK')

    for criteria in ['area','length','height']:
        decrease=BLFwithSequence(source,decrease=criteria)
        if single:  print(criteria,decrease)
        else:
            decrease=np.array(decrease)
            np.savetxt('{}.CSV'.format(criteria),decrease)
            print('{}...OK'.format(criteria))

    # predict=BLFwithSequence(source,seq_path='outputs/0406/fu1500/sequence-0.csv')
    # predict=np.array(predict)
    # np.savetxt('predict.CSV',predict)
    # print('predict...OK')

    ga=BLFwithSequence(source,decrease=None,GA_algo=True)
    if single:  print('GA',ga)
    else:
        ga=np.array(ga)
        np.savetxt('GA.CSV',ga)
        print('GA...OK')

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
    def generateData_fu(poly_num):
        polys=[]
        for i in range(poly_num):
            shape=np.random.randint(0,8) # 矩形 直角三角形 等腰三角形 直角梯形
            b=160
            a=100
            x=a+(b-a)*np.random.random()
            y=a+(b-a)*np.random.random()
            if shape==0 or shape==1:
                poly=np.array([0,0,x,0,x,y,0,y]).reshape(4,2).tolist()
            elif shape==2:
                poly=np.array([0,0,2*x,0,x,y]).reshape(3,2).tolist()
            elif shape==3:
                poly=np.array([0,0,x,y,0,2*y]).reshape(3,2).tolist()
            elif shape==4:
                poly=np.array([0,0,2*x,0,x,y]).reshape(3,2).tolist()
            elif shape==5:
                poly=np.array([0,0,x,y,0,2*y]).reshape(3,2).tolist()
            elif shape==6:
                x2=a+(b-a)*np.random.random()
                poly=np.array([0,0,x2,0,x,y,0,y]).reshape(4,2).tolist()
            elif shape==7:
                y2=a+(b-a)*np.random.random()
                poly=np.array([0,0,x,0,x,y2,0,y]).reshape(4,2).tolist()
            polys.append(poly)
        return polys
    
    @staticmethod
    def generatePolygon(poly_num,max_point_num):
        '''
        随机生成多边形
        poly_num: 多边形个数
        max_point_num: 最大点的个数
        '''
        polys=[]
        for i in range(poly_num):
            point_num=np.random.randint(3,max_point_num+1)
            angle=360/point_num # 根据边数划分角度区域
            theta_start=angle*np.random.random()
            poly=[]
            for j in range(point_num):
                #theta=(theta_start+angle*j)*np.pi/180 # 在每个区域中取角度并转为弧度
                theta_min=angle*j
                theta_max=angle*(j+1)
                theta=(theta_min+(theta_max-theta_min)*np.random.random())*np.pi/180
                #max_r=min(np.math.fabs(500/np.math.cos(theta)),np.math.fabs(500/np.math.sin(theta)))
                r=100+(200-100)*np.random.random()
                x=r*np.math.cos(theta)
                y=r*np.math.sin(theta)
                poly.append([x,y])
            polys.append(poly)
        return polys
    
    @staticmethod
    def generateTestData(dataset_name,size):
        data=[]
        vectors=[]
        for i in tqdm(range(size)):
            # if np.random.random()<0.5:
            #polys=getData()
            # else:
            polys=GenerateData_vector.generatePolygon(10,8)
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
    def xy2vector(source,save_name):
        data=np.load(source,allow_pickle=True)
        vectors=[]
        for index,line in enumerate(tqdm(data)):
            line=line.T
            vector=[]
            for poly in line:
                poly=poly.reshape(4,2).tolist()
                poly=[poly]
                poly=GenerateData_xy.drop0(poly)
                vector.append(vectorFunc(poly[0],cut_nums=128).vector)
            vectors.append(vector)
        vectors=np.array(vectors)
        np.save(save_name,vectors)

    @staticmethod
    def xy2poly(source,save_name):
        data=np.load(source,allow_pickle=True)
        data_new=[]
        for index,line in enumerate(tqdm(data)):
            line=line.T
            line_new=[]
            for poly in line:
                poly=poly.reshape(4,2).tolist()
                line_new.append(poly)
            line_new=GenerateData_xy.drop0(line_new)
            data_new.append(line_new)
        np.save(save_name,data_new)


class GetBestSeq(object):
    def __init__(self,width,polys,criteria='area'):
        self.polys=polys
        self.width=width
        self.criteria=criteria
        
    # 获得面积/长度/高度降序排列的形状结果
    def getDrease(self):
        poly_list=[]
        for poly in self.polys:
            if self.criteria=='length':
                left,bottom,right,top=GeoFunc.checkBoundValue(poly)          
                poly_list.append([poly,right-left])
            elif self.criteria=='height':
                left,bottom,right,top=GeoFunc.checkBoundValue(poly)    
                poly_list.append([poly,top-bottom])
            else:
                poly_list.append([poly,Polygon(poly).area])
        poly_list=sorted(poly_list, key = lambda item:item[1], reverse = True) # 排序，包含index
        # print(poly_list)
        dec_polys=[]
        for item in poly_list:
            dec_polys.append(item[0])
        return dec_polys

    # 从所有的排列中选择出最合适的
    def chooseFromAll(self):
        self.NFPAssistant=NFPAssistant(self.polys)
        all_com=list(itertools.permutations([(i) for i in range(len(self.polys))]))
        min_height=999999999
        best_order=[]
        for item in all_com:
            seq=self.getPolys(item)
            height=BottomLeftFill(self.width,seq,NFPAssistant=self.NFPAssistant).contain_height
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


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn',True) 
    start=time.time()
    #print(GenerateData_vector.generateData_fu(5))
    GenerateData_vector.generateTestData('oct10000',10000)
    getAllNFP('oct10000_xy.npy','oct10000')
    #GenerateData_vector.poly2vector('fu1000_val_xy.npy','fu1000_val')
    #GenerateData_vector.poly2vector('fu1500_xy.npy','fu1500_8')
    #GenerateData_vector.xy2poly('fu1500_val_old.npy','fu1500_val_xy')
    #getBenchmark('poly10000_xy.npy')
    end=time.time()
    print(end-start)
