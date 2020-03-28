'''
    This file is used for region-based quantification
    Convert a polygon into a zero-one matrix
'''
import numpy as np

class geometryCompute(object):
    def __init__(self,aim_polygon):
        self.margin=3 # 多边形的边界
        self.zoom=0.2 # 缩放比例
        self.aim_polygon=aim_polygon
        self.getZOPoly(aim_polygon)

    def getZOPoly(self,aim_polygon):
        '''
        获得zero-one多边形矩阵
        '''
        # 获得多边形的极值
        x_min,x_max,y_min,y_max=self.getExtremum(aim_polygon)
        print("x_min,x_max,y_min,y_max:",x_min,x_max,y_min,y_max)

        #根据margin、extremum和缩放比例重新计算
        new_polygon=self.getNewPolygon(aim_polygon,x_min,y_min)
        height=int((y_max-y_min)*self.zoom+self.margin*2)
        width=int((x_max-x_min)*self.zoom+self.margin*2)
        print("width,height:",width,height)
        
        # 设置初始化的数组
        aim_matrix=np.zeros((height,width))
        print(new_polygon)
        self.contactPolyMatrix(new_polygon,aim_matrix)
        self.fillMatrix(aim_matrix)
        self.displayMatrix(aim_matrix,0,height)

    def contactPolyMatrix(self,aim_polygon,aim_matrix):
        '''
        多边形的边界填充在矩阵中
        '''
        point_nums=len(aim_polygon)
        for index in range(0,point_nums):
            # self.pointFill(aim_matrix,aim_polygon[index])
            if index<point_nums-1:
                self.pointLine(aim_matrix,aim_polygon[index],aim_polygon[index+1])
            elif index==point_nums-1:
                self.pointLine(aim_matrix,aim_polygon[index],aim_polygon[0])
            else:
                continue            

        return aim_matrix
    
    def fillMatrix(self,aim_matrix):
        '''
        填充矩阵内部
        '''
        for i in range(0,len(aim_matrix)):
            aim_list=aim_matrix[i].tolist()
            if aim_list.count(2)==2:
                status=0
                for j in range(0,len(aim_matrix[0])):
                    aim_grid=aim_matrix[i][j]
                    if aim_grid==2 and status==0:
                        status=1
                    elif aim_grid==2 and status==1:
                        status=0
                    elif status==1:
                        aim_matrix[i][j]=1
                    else:
                        continue
                    
        for j in range(0,len(aim_matrix[0])):
            aim_list=aim_matrix[:,j].tolist()
            if aim_list.count(2)==2:
                status=0
                for i in range(0,len(aim_matrix)):
                    aim_grid=aim_matrix[i][j]
                    if aim_grid==2 and status==0:
                        status=1
                    elif aim_grid==2 and status==1:
                        status=0
                    elif status==1:
                        aim_matrix[i][j]=1
                    else:
                        continue

    def pointFill(self,aim_matrix,vertex):
        '''
        把点打在矩阵上
        '''
        x=vertex[0]
        y=vertex[1]
        aim_matrix[int(y),int(x)]=2


    def pointLine(self,aim_matrix,vertex0,vertex1):
        '''
        填充矩阵内部
        '''
        x1=vertex0[0]
        y1=vertex0[1]
        x2=vertex1[0]
        y2=vertex1[1]
        self.pointFill(aim_matrix,vertex0)

        print("x1,y1",int(x1),int(y1),"x2 y2",int(x2),int(y2))
        horizontal=abs(y2-y1)<0.001
        vertical=abs(x2-x1)<0.001
        
        # 水平直线
        if horizontal:
            print("水平直线")
            y=int(y1)
            if int(x1)>int(x2):
                bottom=int(x2)
                top=int(x1)+1
            else:
                bottom=int(x1)
                top=int(x2)+1
            for x in range(bottom,top):
                aim_matrix[y,x]=2
            return True

        # 垂直直线
        if vertical:
            print("垂直直线")
            x=int(x1)
            if int(y1)>int(y2):
                bottom=int(y2)
                top=int(y1)+1
            else:
                bottom=int(y1)
                top=int(y2)+1
            for y in range(bottom,top):
                aim_matrix[y,x]=2
            return True

        # 判断k的范围
        k=(y2-y1)/(x2-x1)
        if abs(k)>1:
            if int(y1)>int(y2):
                bottom=int(y2)
                top=int(y1)+1
            else:
                bottom=int(y1)
                top=int(y2)+1
            for y in range(bottom,top):
                x_aim=(y-y1)/k+x1
                aim_matrix[y,int(x_aim)]=2
        else:
            if int(x1)>int(x2):
                bottom=int(x2)
                top=int(x1)+1
            else:
                bottom=int(x1)
                top=int(x2)+1
            for x in range(bottom,top):
                y_aim=(x-x1)*k+y1
                aim_matrix[int(y_aim),x]=2
        
        return aim_matrix         
        
    def getExtremum(self,aim_polygon):
        '''
        获得多边形的最高/低/左/右点
        '''
        x_max=y_max=0
        x_min=y_min=100000000
        for item in aim_polygon:
            if item[0]>x_max:
                x_max=item[0]  
            if item[1]>y_max:
                y_max=item[1]
            if item[0]<x_min:
                x_min=item[0]  
            if item[1]<y_min:
                y_min=item[1]
        return int(x_min),int(x_max),int(y_min),int(y_max)
    

    def getNewPolygon(self,aim_polygon,x_min,y_min):
        '''
        根据最大最小值以及比例计算
        '''
        new_polygon=[]
        for vertex in aim_polygon:
            vertex[0]=(vertex[0]-x_min)*self.zoom+self.margin
            vertex[1]=(vertex[1]-y_min)*self.zoom+self.margin
            new_polygon.append(vertex)
        return new_polygon
    
    def displayMatrix(self,matrix,start,end):
        '''
        显示最终结果
        '''
        for i in range(start,end):
            for x in np.nditer(matrix[i]):
                if x==2 or x==1:
                    print (int(x), end=" , " )
                else:
                    print ("  ", end=", " )
            print("\n")


if __name__ == '__main__':
    gc=geometryCompute([[0,0],[100,100],[50,100]])
