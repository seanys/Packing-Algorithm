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
#include "plot.cpp"
#include <time.h>

class BLF{
protected:
    int polygons=0; // 输入形状可以考虑对应的序列
    PolysArrange polys_arrange; // 形状的排样情况
    NFPAssistant *nfp_assistant;
public:
    BLF(){
        DataAssistant::readData(polys_arrange); // 加载数据
        nfp_assistant=new NFPAssistant("/Users/sean/Documents/Projects/Data/fu.csv",polys_arrange.type_num,4);
        
    };
    void run(){
        Polygon poly1,poly2;
        read_wkt("POLYGON((480 200,480 380,200 380,200 760,1000 760,1000 200,480 200))", poly1);
        read_wkt("POLYGON((0 200,480 200,480 580,0 580,0 200))", poly2);
        list<Polygon> output;
        boost::geometry::intersection(poly1, poly2, output);
        for(auto item:output){
            cout<<"output:"<<dsv(item)<<endl;
        };
//        clock_t start,end;
//        start=clock();

//        placeFirstPoly();
//        for(int j=1;j<3;j++){
//            placeNextPoly(j);
//        }
        
//        end=clock();
//
//        cout<<"运行总时间"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;

    };
    void placeFirstPoly(){
        VectorPoints ifr;
        PackingAssistant::getIFR(polys_arrange.polys[0], polys_arrange.width, 99999999, ifr);
        PackingAssistant::slideToPosition(polys_arrange.polys[0], ifr[1]);
        PrintAssistant::print2DVector(polys_arrange.polys[0], true);
    };
    void placeNextPoly(int j){
        // 获得IFR
        VectorPoints ifr;
        PackingAssistant::getIFR(polys_arrange.polys[0], polys_arrange.width, 99999999, ifr);
        // IFR转Polygon并计算差集
        Polygon IFR;
        GeometryProcess::convertPoly(ifr,IFR);
        list<Polygon> feasible_region={IFR};
        cout<<endl<<"IFR:"<<dsv(IFR)<<endl;
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
//            cout<<"nfp:"<<dsv(nfp_poly)<<endl;
            PolygonsOperator::polysDifference(feasible_region,nfp_poly);
            
        };
        // 遍历获得所有的点
        VectorPoints all_points;
        GeometryProcess::getAllPoints(feasible_region,all_points);
        // 选择最左侧的点
        vector<double> bl_point;
//        PrintAssistant::print2DVector(all_points,true);
        PackingAssistant::getBottomLeft(all_points,bl_point);
//        PrintAssistant::print1DVector(bl_point,true);
        PackingAssistant::slideToPosition(polys_arrange.polys[j],bl_point);
        PrintAssistant::print2DVector(polys_arrange.polys[j], true);
    }
};

class LPSearch{
protected:
    DataAssistant *data_assistant;
    int polygons; // 全部的形状
    vector<vector <double>> overlap; // 重叠情况
    PltFunc *plt_func;
public:
    LPSearch(){
        polygons=0;
        data_assistant=new DataAssistant();
    }
    double run(){
        cout<<"success"<<endl;
        Polygon Poly;
        VectorPoints poly={{1, 2},{2,4},{0,5}};
//        GeometryProcess::convertPoly(poly,Poly);
        
        VectorPoints border_points;
        PackingAssistant::getBorder(poly,border_points);
//        PolygonFunctions::polyBoundPoints();
//        data_assistant->test();
//        plt_func->pltTest();
//        geo_func->getIntersection();
//        geo_func->polysUnion();
        return 0.0;
    }
};

int main(int argc, const char * argv[]) {
//    LPSearch *lp;
//    lp=new LPSearch();
//    lp->run();

    
    BLF *blf;
    blf=new BLF();
    blf->run();
    
    return 0;
}

