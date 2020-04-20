//
//  main.cpp
//  Nesting Problem
//
//  Created by Yang Shan on 2020/4/13.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//

#include <iostream>
#include <vector>
#include <stdbool.h>
#include <csv/reader.hpp>
#include "geometry.cpp"
//#include "plot.cpp"
#include <time.h>

class BLF{
protected:
    int polygons=0; // 输入形状可以考虑对应的序列
    PolysArrange polys_arrange; // 形状的排样情况
    NFPAssistant *nfp_assistant;
public:
    BLF(){
        DataAssistant::readData(polys_arrange); // 加载数据
        nfp_assistant=new NFPAssistant("/Users/sean/Documents/Projects/Data/fu_clock.csv",polys_arrange.type_num,4);
        
    };
    void run(){
        placeFirstPoly();
        int _max=int(polys_arrange.polys.size());
        for(int j=1;j<_max;j++){
            placeNextPoly(j);
        }
    };
    void placeFirstPoly(){
        VectorPoints ifr;
        PackingAssistant::getIFR(polys_arrange.polys[0], polys_arrange.width, 99999999, ifr);
        PackingAssistant::slideToPosition(polys_arrange.polys[0], ifr[3]);
        PrintAssistant::print2DVector(polys_arrange.polys[0], true);
    };
    void placeNextPoly(int j){
        // 获得IFR
        VectorPoints ifr;
        PackingAssistant::getIFR(polys_arrange.polys[j], polys_arrange.width, 99999999, ifr);
        // IFR转Polygon并计算差集
        Polygon IFR;
        GeometryProcess::convertPoly(ifr,IFR);
        list<Polygon> feasible_region={IFR};
//        cout<<endl<<"IFR:"<<dsv(IFR)<<endl;
        // 逐一计算
        for(int i=0;i<j;i++){
            // 初步处理NFP多边形
            VectorPoints nfp;
            int type_i=polys_arrange.polys_type[i],type_j=polys_arrange.polys_type[j];
            int oi=polys_arrange.polys_orientation[i],oj=polys_arrange.polys_orientation[j];
            nfp_assistant->getNFP(type_i,type_j,oi,oj,polys_arrange.polys[i],nfp);
            // 转化为多边形并求交集
            Polygon nfp_poly;
            GeometryProcess::convertPoly(nfp,nfp_poly);
            PolygonsOperator::polysDifference(feasible_region,nfp_poly);
            
        };
        // 遍历获得所有的点
        VectorPoints all_points;
        GeometryProcess::getAllPoints(feasible_region,all_points);
        // 选择最左侧的点
        vector<double> bl_point;
        PackingAssistant::getBottomLeft(all_points,bl_point);
        PackingAssistant::slideToPosition(polys_arrange.polys[j],bl_point);
        PrintAssistant::print2DVector(polys_arrange.polys[j], true);
    }
};

class LPSearch{
protected:
    VectorPoints cur_ifr; // 获得当前的IFR
    double poly_num;
    vector<VectorPoints> all_nfps; // 获得当前全部NFP
    vector<vector<vector<double>>> edges_target_funs,points_target_funs; // 分别获得直线和点的目标函数
    
    PolysArrange best_solution,cur_solution; // 最佳解和当前解，不需要用指针
    vector<VectorPoints> best_polys,polys; // 暂存当前的全部形状
    vector<int> cur_orientation; // 当前采用的方向
    vector<int> polys_type; // 所有形状情况
    double best_length,cur_length,width; // 长度等性质

    vector<vector <double>> poly_overlap; // 重叠情况
    vector<vector <double>> nfp_overlap_pair; // NFP的重叠情况
    vector<vector <double>> miu; // 重叠情况
    vector<vector <VectorPoints>> all_polygons; // 全部形状的加载
    
    int choosed_index; // 当前选择的index
        
    NFPAssistant *nfp_assistant; // NFP辅助操作
public:
    LPSearch(){
        DataAssistant::readBLF(1,best_solution); // 读取BLF的解
        cur_solution=best_solution; // 赋值当前解
        nfp_assistant=new NFPAssistant("/Users/sean/Documents/Projects/Data/fu_clock.csv",best_solution.type_num,4); // NFP辅助

        cur_orientation=cur_solution.polys_orientation; // 赋值全部形状（变化）
        cur_length=cur_solution.length; // 当前长度（变化）
        best_length=best_solution.length; // 最佳长度（变化）

        width=best_solution.width; // 宽度（不变）
        poly_num=best_solution.total_num; // 形状总数（不变）
        polys_type=cur_solution.polys_type; // 形状类别（不变）
    };
    void run(){
        cout<<"success"<<endl;
        Polygon Poly;
        
    };
    // 输入序列形状，每次处理一个序列
    void minimizeOverlap(){
        initial2DVector(1,poly_num,miu); // 初始化Miu的值
        
    };
    
    // 获得某个形状的全部NFP和IFR
    void getNFPIFR(int j){
        // 获得全部的NFP
        all_nfps={};
        for(int i=0;i<poly_num;i++){
            // 如果等于则直接加空的
            if(i==j){
                all_nfps.push_back({});
                continue;
            }
            // 获得对应的NFP
            VectorPoints nfp;
            nfp_assistant->getNFP(polys_type[i], polys_type[j], cur_orientation[i], cur_orientation[j], polys[j], nfp);
            all_nfps.push_back(nfp);
        }
        // 获得IFR形状
        cur_ifr={};
        PackingAssistant::getIFR(cur_solution.polys[j], width, cur_length, cur_ifr);
    };
    
    // 在获得NFP和IFR之后获得目所有目标函数
    void getTargetFunc(){
        for(int i=0;i<poly_num;i++){
            // 初始化添加
            edges_target_funs.push_back({});
            points_target_funs.push_back({});
            // 如果相同则跳过
            if(i==choosed_index){
                continue;
            }
            // 首先计算全部的直线
            vector<VectorPoints> all_edges;
            getAllEdges(all_nfps[i], all_edges);
            for(auto edge:all_edges){
                vector<double> coeff;
                getEdgeCoeff(coeff,edge);
                edges_target_funs[i].push_back(coeff);
            }
            // 其次遍历所有的点（其实不用封装也OK）
            for(auto pt:all_nfps[i]){
                points_target_funs[i].push_back({pt[0],pt[1]});
            }
        }
    };
    
    // 获得某个位置的最低的Depth（点可以直接遍历所有的点即可）
    void getPolyDepth(vector<double> pt){
        
    };
    
    // 更新权重参数Miu
    void updateMiu(){
        // 寻找最大的重叠
        double _max=0;
        for(auto line:poly_overlap){
            for(auto item:line){
                if(item>_max){
                    _max=item;
                }
            }
        }
        // 更新全部的Miu
        for(int i=0;i<poly_num;i++){
            for(int j=0;j<poly_num;j++){
                miu[i][j]+=poly_overlap[i][j]/_max;
                miu[j][i]+=poly_overlap[j][i]/_max;
            }
        }
    };
    
    // 获得最小的Penetration Depth的位置
    void searchForBestPosition(){
        
    };
    
    // 获得NFP的重叠情况
    void getNFPOverlapPair(){
        for(int i=0;i<poly_num;i++){
            nfp_overlap_pair.push_back({});
            for(int j=0;j<poly_num;j++){
                // 如果是选择index或者相同，全部是1
                if(i==j||i==choosed_index||j==choosed_index){
                    nfp_overlap_pair[i].push_back(0);
                    continue;
                }
                if(PackingAssistant::judgeOverlap(all_nfps[i],all_nfps[j])==true){
                    nfp_overlap_pair[i].push_back(1);
                }else{
                    nfp_overlap_pair[i].push_back(0);
                }
            }
        }
    };
    
    // 增加获得全部重叠，用于比较情况
    double getTotalOverlap(){
        double total_overlap=0;
        for(int i=0;i<poly_num-1;i++){
            for(int j=i+1;j<poly_num;j++){
                if(i==j){
                    continue;
                }
                total_overlap+=PackingAssistant::overlapArea(polys[i],polys[j]);
            }
        }
        return total_overlap;
    };
    
    // 初始化当前的整个重叠（两两间的重叠）
    void initialPolyOverlap(){
        initial2DVector(0,poly_num,poly_overlap);
        for(int i=0;i<poly_num-1;i++){
            for(int j=i+1;j<poly_num;j++){
                double overlap=PackingAssistant::overlapArea(polys[i], polys[j]);
                poly_overlap[i][j]=overlap;
                poly_overlap[j][i]=overlap;
            }
        }
    };
    
    // 更新多边形的重叠情况（单独更新某个形状）
    void updatePolyOverlap(int i){
        for(int j=0;j<poly_num;j++){
            if(i==j){
                continue;
            }
            double overlap=PackingAssistant::overlapArea(polys[i], polys[j]);
            poly_overlap[j][i]=overlap;
            poly_overlap[i][j]=overlap;
        }
    };
    
    // 平移多边形到内部
    void slideToContainer(){
        
    };
    
    /*
     以下函数可以封装到PackingAssistant中！
     */
    
    // 判断当前解是否可行（获得重叠）
    bool judgeFeasible(){
        for(int i=0;i<poly_num-1;i++){
            for(int j=i+1;j<poly_num;j++){
                if(poly_overlap[i][j]>0){
                    return false;
                }
            }
        }
        return true;
    };
    
    // 获得某个多边形所有的边
    void getAllEdges(VectorPoints poly,vector<VectorPoints> &all_edges){
        for(int i=0;i<poly_num;i++){
            if(i==poly_num-1){
                all_edges.push_back({poly[i],poly[0]});
            }else{
                all_edges.push_back({poly[i],poly[i+1]});
            }
        }
    };
    
    // 获得点到直线的距离的系数参数
    void getEdgeCoeff(vector<double> coeff, VectorPoints edge){
        double A=edge[0][1]-edge[1][1];
        double B=edge[1][0]-edge[0][0];
        double C=edge[0][0]*edge[1][1]-edge[1][0]*edge[0][1];
        double D=sqrt(A*A+B*B);
        coeff={A/D,B/D,C/D};
    };
    
    // 初始化二维数组
    void initial2DVector(double initial_value,double initial_size,vector<vector<double>> target){
        target={};
        for(int i=0;i<initial_size;i++){
            target.push_back({});
            for(int j=0;j<initial_size;j++){
                target[i].push_back(initial_value);
            }
        }
    };
    
};

int main(int argc, const char * argv[]) {
    LPSearch *lp;
    lp=new LPSearch();
    lp->run();

    
//    BLF *blf;
//    blf=new BLF();
//    blf->run();
//        clock_t start,end;
//        start=clock();
//        end=clock();
//        cout<<"运行总时间"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;

    return 0;
}

