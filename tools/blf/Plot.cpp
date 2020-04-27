//
//  Plot.cpp
//  Nesting Problem
//
//  Created by Yang Shan on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//

#include <iostream>
#include <csv/writer.hpp>
#include <vector>

using namespace std;

class FileAssistant{
public:
    /*
     用于记录形状，输入polygons，记录到目标文件
     */
    static void recordPolys(string _path,vector<vector<vector<double>>> all_polys){
        // 读取并设置文件头
        csv::Writer foo(_path);
        foo.configure_dialect()
        .delimiter(", ")
        .column_names("polygon");
        
        // 将形状全部写入
        for (long i = 0; i < all_polys.size(); i++) {
            string res=vectorToString(all_polys[i]);
            foo.write_row(res);
        }
        foo.close();
    };
    /*
     
     */
    static string vectorToString(vector<vector<double>> poly){
        string str_poly="\"[";
        for(int i=0;i<poly.size();i++){
            str_poly=str_poly+"[";
            str_poly=str_poly+to_string(poly[i][0]);
            str_poly=str_poly+",";
            str_poly=str_poly+to_string(poly[i][1]);
            str_poly+="]";
            if(i<poly.size()-1){
                str_poly+=",";
            }
        }
        str_poly=str_poly+"]\"";
        return str_poly;
    }
};

class PltFunc{
public:
    static void pltShow(vector<vector<vector<double>>> all_polys){
        string _path="/Users/sean/Documents/Projects/Packing-Algorithm/record/lp_result.csv";
        FileAssistant::recordPolys(_path,all_polys);
        system("/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7 /Users/sean/Documents/Projects/Packing-Algorithm/new_data.py");
    }
};
