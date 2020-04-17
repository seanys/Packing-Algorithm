//
//  main.pch
//  Nesting Problem
//
//  Created by 爱学习的兔子 on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//


#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <string>

using namespace std;

// 形状排样的结果
struct PolysArrange{
    vector<vector<vector<double>>> polys; // 所有形状的情况
    vector<vector<double>> polys_position; // 所有形状的顶点位置
    vector<double> polys_orientation; // 所有形状的方向
    double width;
    double total_area;
};

// NFP的存储对象——两两组合，四个方向
struct AllNFP{
    vector<vector<vector<double>>> polys; // 所有形状的情况
    vector<vector<double>> polys_position; // 所有形状的顶点位置
    vector<double> polys_orientation; // 所有形状的方向
    double width;
    double total_area;
};

// 输出数组的函数集合
class ProcessFunc{
public:
    /*
    输出一维数组——泛型，暂时统一用double，比如Orientation
    */
    template <typename T>
    static void print1DVector (vector<T> &vec, bool with_endle)
    {
        vector<double>::iterator ite = vec.begin();
        cout<<"[";
        for (; ite != vec.end(); ite++){
            cout << *ite << ",";
        }
        cout<<"],";
        if(with_endle==true){
            cout<<endl;
        }
    };
    /*
    输出二维数组——泛型，暂时统一用double，如Positions
    */
    template <typename T>
    static void print2DVector (vector<vector<T>> &vec,bool with_endle)
    {
        vector<vector<double>>::iterator ite = vec.begin();
        cout<<"[";
        for (; ite != vec.end(); ite++){
            print1DVector(*ite,false);
        }
        cout<<"]";
        if(with_endle==true){
            cout<<endl;
        }
    };
    /*
    输出三维数组——泛型，暂时统一用double，主要是Polygons
    */
    template <typename T>
    static void print3DVector (vector<vector<vector<T>>> &vec,bool with_endle)
    {
        vector<vector<vector <double>>>::iterator ite = vec.begin();
        cout<<"[";
        for (; ite != vec.end(); ite++){
            print2DVector(*ite,false);
        }
        cout<<"]";
        if(with_endle==true){
            cout<<endl;
        }
    }
};
