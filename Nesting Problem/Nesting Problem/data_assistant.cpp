//
//  Data Assistant.cpp
//  Nesting Problem
//
//  Created by 爱学习的兔子 on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//
//  CSV Kits
//  https://github.com/p-ranav/csv
// 

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <csv/reader.hpp>

using namespace std;


class DataAssistant{
public:
    double readCSV(){
//        csv::Reader foo;
//        foo.read("/Users/sean/Documents/Projects/Packing-Algorithm/record/c_blf.csv");
//        while(foo.busy()) {
//          if (foo.ready()) {
//            auto row = foo.next_row();  // Each row is a csv::unordered_flat_map (github.com/martinus/robin-hood-hashing)
//            auto total_area = row["total_area"];       // You can use it just like an std::unordered_map
//            auto orientation = row["orientation"];
//            cout<<orientation<<endl;
//          }
//        }
        return 0;
    };
    // 一维字符串转数组
    vector<double> stringToVector(string line){
        vector<double> res;
        return res;
    }
    // 二维字符串转数组
//    vector<vector<double>> stringToVectorTwo(string line){
//        vector<double> lineArray;
//        return
//    };
};

