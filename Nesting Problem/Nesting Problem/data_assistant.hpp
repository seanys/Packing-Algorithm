//
//  data_assistant.hpp
//  Nesting Problem
//
//  Created by Yang Shan on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.

#include <iostream>
#include <string>
#include <vector>
#include <iterator>

using namespace std;

// 基础定义
typedef vector<vector<double>> VectorPoints;

// 形状排样的结果
struct PolysArrange{
    string name=""; // 该数据集的来源
    int type_num=0; // 形状类别总数
    int total_num=0; // 总形状数目
    double width; // 宽度
    double total_area; // 总面积
    vector<vector<vector<double>>> polys; // 所有形状的情况
    vector<double> polys_type; // 形状对应的关系
    vector<vector<double>> polys_position; // 所有形状的顶点位置
    vector<double> polys_orientation; // 所有形状的方向
};

// 输出数组的函数集合
class PrintAssistant{
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
