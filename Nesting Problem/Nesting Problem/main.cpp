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
#include "data_assistant.cpp"
#include "geometry.cpp"
#include "plot.cpp"

using namespace std;


class LPSearch{
protected:
    DataAssistant *data_assistant;
    int polygons; // 全部的形状
    vector<vector <double>> overlap; // 重叠情况
    PltFunc *plt_func;
    GeoFunc *geo_func;
public:
    LPSearch(){
        polygons=0;
        data_assistant=new DataAssistant();
    }
    double run(){
        cout<<"success"<<endl;
//        data_assistant->readCSV();
        plt_func->pltTest();
//        geo_func->getIntersection();
//        geo_func->polysUnion();
        return 0.0;
    }
};

int main(int argc, const char * argv[]) {
    std::cout << "Hello, World!\n";
    LPSearch *lp;
    lp=new LPSearch();
    lp->run();
    
    

    return 0;
}

