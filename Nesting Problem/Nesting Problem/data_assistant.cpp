//
//  Data Assistant.cpp
//  Nesting Problem
//
//  Created by Sean Yang on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//
//  CSV Kits
//  https://github.com/p-ranav/csv
// 

#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <csv/reader.hpp>
#include <csv/writer.hpp>
#include <x2struct/x2struct.hpp>
#include "main.hpp"

using namespace std;
using namespace x2struct;



class DataAssistant{
public:
    void test(){
        PolysArrange blf_result;
        readBLF(1,blf_result);
    };
    /*
     直接读取指定的Index
     */
    static void readCSV(){
        csv::Reader foo;
        foo.read("/Users/sean/Documents/Projects/Packing-Algorithm/record/c_blf.csv");
        auto rows = foo.rows();
        cout<<rows[1]["orientation"]<<endl;
    };
    /*
     读取BLF某一行的结果，包括宽度、方向、总面积、位置和形状
     */
    static void readBLF(int index,PolysArrange blf_result){
        csv::Reader foo;
        foo.read("/Users/sean/Documents/Projects/Packing-Algorithm/record/c_blf.csv");
        auto rows = foo.rows();
        blf_result.width=stod(rows[index]["width"]);
        blf_result.total_area=stod(rows[index]["total_area"]);
        load1DVector(rows[index]["polys_orientation"],blf_result.polys_orientation);
        load2DVector(rows[index]["polys_position"],blf_result.polys_position);
        load3DVector(rows[index]["polys"],blf_result.polys);
    };
    /*
     加载一维的数组向量并返回
     */
    template <typename T>
    static void load1DVector(string str,vector<T> &vec){
        str.erase(remove(str.begin(), str.end(), '\"'), str.end());
        X::loadjson(str, vec, false);
    };
    /*
     加载二维数组
     */
    template <typename T>
    static void load2DVector(string str,vector<vector<T>> &vec){
        str.erase(remove(str.begin(), str.end(), '\"'), str.end());
        X::loadjson(str, vec, false);
    };
    /*
     加载三维数组
     */
    template <typename T>
    static void load3DVector(string str,vector<vector<vector<T>>> &vec){
        str.erase(remove(str.begin(), str.end(), '\"'), str.end());
        X::loadjson(str, vec, false);
    }
};

class WriterAssistant{
public:
    /*
     写入CSV
     */
    static void writeCSV(){
        csv::Writer foo("/Users/sean/Documents/Projects/Packing-Algorithm/record/test.csv");
        foo.configure_dialect()
          .delimiter(", ")
          .column_names("a", "b", "c");
        
        // 需要现转化为String再写入
        for (long i = 0; i < 3000000; i++) {
            foo.write_row("1", "2", "3");                                     // parameter packing
            foo.write_row(map<string, string>{
              {"a", "7"}, {"b", "8"}, {"c", "9"} });
        }
        foo.close();
    };
}
