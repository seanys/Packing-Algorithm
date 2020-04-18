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


class BLF{
protected:
    int polygons=0; // 输入形状可以考虑对应的序列
    PolysArrange polys_arrange; // 形状的排样情况
    NFPAssistant nfp_assistant=NFPAssistant("/Users/sean/Documents/Projects/Data/fu.csv",polys_arrange.type_num,4);
public:
    BLF(){
        DataAssistant::readData(polys_arrange); // 加载数据
//        memcpy(&initial_arrange, &arrange_result, sizeof(initial_arrange)); // 复制到排样结果
//        arrange_result.polys={}; // 需要定义为空
    };
    void run(){
        placeFirstPoly();
        for(int j=1;j<polys_arrange.total_num;j++){
            placeNextPoly(j);
        }
    };
    void placeFirstPoly(){
        VectorPoints IFR;
        PackingAssistant::getIFR(polys_arrange.polys[0], polys_arrange.width, 99999999, IFR);
        PackingAssistant::slideToPosition(polys_arrange.polys[0], IFR[1]);
    };
    void placeNextPoly(int j){
        for(int i=0;i<j;i++){
            
        };
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
    std::cout << "Hello, World!\n";
//    LPSearch *lp;
//    lp=new LPSearch();
//    lp->run();
    
    BLF *blf;
    blf=new BLF();
    blf->run();

    return 0;
}

