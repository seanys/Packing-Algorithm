//
//  tools.cpp
//  Simple LP Search
//
//  Created by 爱学习的兔子 on 2020/7/5.
//  Copyright © 2020 Yang Shan. All rights reserved.
//

/* 函数包括
 1. 基础定义模块
 2. 文件读写模块
 3. 排样基础模块
 */


#include <iostream>
#include <csv/writer.hpp>
#include <vector>
#include <algorithm>
#include <csv/reader.hpp>
#include <x2struct/x2struct.hpp>
#include <assert.h>

#define BIAS 0.000001

using namespace std;
using namespace x2struct;

typedef vector<vector<double>> Polygon;

class TOOLS{
public:
    // 去除分号
    static string removeSem(string _str){
        string new_str = _str;
        new_str.erase(remove(new_str.begin(), new_str.end(), '\"' ),new_str.end());
        return new_str;
    };
    // 将形状转化为string格式进一步处理
    static string vectorToString(vector<vector<double>> poly){
        string str_poly = "\"[";
        for(int i = 0; i < poly.size(); i++){
            str_poly = str_poly + "[";
            str_poly = str_poly + to_string(poly[i][0]);
            str_poly = str_poly + ",";
            str_poly = str_poly + to_string(poly[i][1]);
            str_poly += "]";
            if(i < poly.size() - 1){
                str_poly += ",";
            }
        }
        str_poly = str_poly+"]\"";
        return str_poly;
    }
    // 打印一维度数组
    template <typename T>
    static void print1DVector (vector<T> &vec, bool with_endle)
    {
        cout<<"[";
        for (int i = 0; i < vec.size(); i++){
            cout << vec[i];
            if(i<(int)vec.size() - 1){
                cout << ",";
            }
        }
        cout << "]";
        if(with_endle == true){
            cout << endl;
        }
    };

    // 输出二维数组——泛型，暂时统一用double，如Positions
    template <typename T>
    static void print2DVector (vector<vector<T>> &vec,bool with_endle)
    {
        cout << "[";
        for (int i = 0; i < vec.size(); i++){
            print1DVector(vec[i], false);
            if(i < (int)vec.size() - 1){
                cout << ",";
            }
        }
        cout << "]";
        if(with_endle == true){
            cout << endl;
        }
    };

    // 输出三维数组——泛型，暂时统一用double，主要是Polygons
    template <typename T>
    static void print3DVector (vector<vector<vector<T>>> &vec,bool with_endle)
    {
        cout << "[";
        for (int i = 0;i<vec.size();i++){
            print2DVector(vec[i],false);
        }
        cout<<"]";
        if(with_endle==true){
            cout<<endl;
        }
    }

    // 增加布丁长度的向量
    void pushVector(Polygon poly,Polygon new_poly){
        for(auto pt:new_poly){
            poly.push_back(vector<double> {pt[0],pt[1]});
        }
        print2DVector(poly,false);
    };

    // 加载一维的数组向量并返回
    template <typename T>
    static void load1DVector(string str,vector<T> &vec){
        str.erase(remove(str.begin(), str.end(), '\"'), str.end());
        X::loadjson(str, vec, false);
    };
        
    // 加载二维数组
    template <typename T>
    static void load2DVector(string str,vector<vector<T>> &vec){
        str.erase(remove(str.begin(), str.end(), '\"'), str.end());
        X::loadjson(str, vec, false);
    };
        
    // 加载三维数组
    template <typename T>
    static void load3DVector(string str,vector<vector<vector<T>>> &vec){
        str.erase(remove(str.begin(), str.end(), '\"'), str.end());
        X::loadjson(str, vec, false);
    };
        
    // List数组的增长
    template <typename T>
    static void appendList(list<T> &old_list,list<T> &new_list){
        for(auto item:new_list){
            old_list.push_back(item);
        }
    };
    // 初始化一个多维向量
    static void initial3DVector(int _size, double target_val, vector<vector<double>> &target_list){
        target_list.clear(); // 部分情况下是不一定为空的
        for(int i = 0; i < _size; i++){
            target_list.push_back({});
            for(int j = 0; j < _size; j++){
                target_list[i].push_back(target_val);
            }
        }
    };
    
};

/*
 CSV的读写于记录
 */
class CSVAssistant{
public:
    // 将形状输出到目标的文件夹
    static void recordPoly(string _path,vector<vector<vector<double>>> all_polys){
        // 读取并设置文件头
        csv::Writer foo(_path);
        foo.configure_dialect()
        .delimiter(", ")
        .column_names("polygon");
        
        // 将形状全部写入
        for (long i = 0; i < all_polys.size(); i++) {
            string res = TOOLS::vectorToString(all_polys[i]);
            foo.write_row(res);
        }
        foo.close();
    };
    // 根据形状结果是否成功，将其存储到目标的文件
    static void recordResult(string _path,vector<vector<vector<double>>> all_polys){
        // 读取并设置文件头
        csv::Writer foo(_path);
        foo.configure_dialect()
        .delimiter(", ")
        .column_names("polygon");
        
        // 将形状全部写入
        for (long i = 0; i < all_polys.size(); i++) {
            string res = TOOLS::vectorToString(all_polys[i]);
            foo.write_row(res);
        }
        foo.close();
    };
    // 读取CSV文件
    static void readCSV(){
        csv::Reader foo;
        foo.read("/Users/sean/Documents/Projects/Packing-Algorithm/record/c_blf.csv");
        auto rows = foo.rows();
        cout<<rows[1]["orientation"]<<endl;
    };
    // 写文件
    static void writeCSV(){
        csv::Writer foo("/Users/sean/Documents/Projects/Packing-Algorithm/record/test.csv");
        // 设置头文件
        foo.configure_dialect()
          .delimiter(", ")
          .column_names("a", "b", "c");
        
        // 需要现转化为String再写入
        for (long i = 0; i < 10; i++) {
            // 可以直接写入
            foo.write_row("1", "2", "3");
            // 也可以按mapping写入
            foo.write_row(map<string, string>{
              {"a", "7"}, {"b", "8"}, {"c", "9"} });
        }
        foo.close();
    };
};

/*
 展示部分形状或单个形状
 */
class PltFunc{
private:
    vector<vector<vector<double>>> polys; // 全部的形状
    vector<string> colors; // 形状的颜色
public:
    // 后续修改为初始化，可以增加形状
    PltFunc(){
        polys = {};
        colors = {};
    };
    // 增加一个形状
    void addPolygon(vector<vector<double>> poly,string color){
        polys.push_back(poly);
        colors.push_back(color);
    };
    // 多个形状的加载
    static void polysShow(vector<vector<vector<double>>> all_polys){
        string _path = "/Users/sean/Documents/Projects/Packing-Algorithm/record/lp_result.csv";
        system("/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7 /Users/sean/Documents/Projects/Packing-Algorithm/new_data.py");
    };
    // 单个形状的加载
    static void polyShow(vector<vector<double>> poly){
        vector<vector<vector<double>>> all_polys = {poly};
        polysShow(all_polys);
    }
};


// 包含处理函数
class PackingAssistant{
public:
    // 标准化数据（扩大或者缩小倍数）
    static void normData(Polygon *poly,double multiplier){
        for(int i=0; i<(*poly).size();i++){
            (*poly)[i][0]=(*poly)[i][0]*multiplier;
            (*poly)[i][1]=(*poly)[i][1]*multiplier;
        }
    };

    // 获得Inner Fit Rectangle
    static void getIFR(vector<vector<double>> polygon,double container_width,double container_length,Polygon &IFR){
        // 初始参数，获得多边形特征
        Polygon border_points;
        getBorder(polygon,border_points);
                
        double poly_width_left = border_points[3][0]-border_points[0][0];
        double poly_width_right = border_points[2][0]-border_points[3][0];
        double poly_height = border_points[3][1]-border_points[1][1];

        // IFR具体计算（从左上角顺时针计算）
        IFR.push_back({poly_width_left,container_width});
        IFR.push_back({container_length-poly_width_right,container_width});
        IFR.push_back({container_length-poly_width_right,poly_height});
        IFR.push_back({poly_width_left,poly_height});
    };
    // 移动某个多边形
    static void slidePoly(Polygon &polygon,double delta_x,double delta_y){
        for(int i=0;i<polygon.size();i++){
            polygon[i][0] = polygon[i][0]+delta_x;
            polygon[i][1] = polygon[i][1]+delta_y;
        }
    };
    // 移动多边形到某个位置（参考点）
    static void slideToPosition(Polygon &polygon,vector<double> target_pt){
        vector<double> refer_pt;
        getTopPt(polygon,refer_pt);
        double delta_x = target_pt[0]-refer_pt[0];
        double delta_y = target_pt[1]-refer_pt[1];
        for(int i=0;i<polygon.size();i++){
            polygon[i][0] = polygon[i][0] + delta_x;
            polygon[i][1] = polygon[i][1] + delta_y;
        }
    };
    // 获得多边形的所有的边界情况min_x min_y max_x max_y
    static void getBound(Polygon polygon,vector<double> &bound){
        Polygon border_points;
        getBorder(polygon,border_points);
        bound={border_points[0][0],border_points[1][1],border_points[2][0],border_points[3][1]};
    };
    // 遍历获得一个多边形的最左侧点
    static void getBottomLeft(Polygon polygon,vector<double> &bl_point){
        bl_point={999999999,999999999};
        for(auto point:polygon){
            if(point[0]<bl_point[0] || (point[0]==bl_point[0]&&point[1]<bl_point[1]) ){
                bl_point[0]=point[0];
                bl_point[1]=point[1];
            }
        };
    };
    // 仅仅获得最右侧点，同样为逆时针处理（用于判断是逗超出界限）
    static void getRightPt(Polygon polygon,vector<double> &right_pt){
        right_pt={-9999999999,0};
        int poly_size=(int)polygon.size();
        for(int i=poly_size-1;i>=0;i--){
            if(polygon[i][0]>right_pt[0]){
                right_pt[0]=polygon[i][0];
                right_pt[1]=polygon[i][1];
            }
        }
    };
    // 获得多边形参考点
    static void getTopPt(Polygon polygon,vector<double> &refer_pt){
        refer_pt={0,-9999999999};
        int poly_size=(int)polygon.size();
        for(int i=poly_size-1;i>=0;i--){
            if(polygon[i][1] > refer_pt[1]){
                refer_pt[0] = polygon[i][0];
                refer_pt[1] = polygon[i][1];
            }
        }
    };
    // 获得多边形的底部的点
    static void getBottomPt(Polygon polygon,vector<double> &bottom_pt){
        bottom_pt={0,9999999999};
        int poly_size=(int)polygon.size();
        for(int i=poly_size-1;i>=0;i--){
            if(polygon[i][1]<bottom_pt[1]){
                bottom_pt[0]=polygon[i][0];
                bottom_pt[1]=polygon[i][1];
            }
        }
    };
    
    // 获得多边形的边界点
    static void getBorder(Polygon polygon,Polygon &border_points){
        // 增加边界的几个点
        border_points.push_back(vector<double>{9999999999,0});
        border_points.push_back(vector<double>{0,999999999});
        border_points.push_back(vector<double>{-999999999,0});
        border_points.push_back(vector<double>{0,-999999999});
        // 遍历所有的点，分别判断是否超出界限
        int poly_size=(int)polygon.size();
        for(int i=poly_size-1;i>=0;i--){
            // 左侧点判断
            if(polygon[i][0]<border_points[0][0]){
                border_points[0][0]=polygon[i][0];
                border_points[0][1]=polygon[i][1];
            }
            // 下侧点判断
            if(polygon[i][1]<border_points[1][1]){
                border_points[1][0]=polygon[i][0];
                border_points[1][1]=polygon[i][1];
            }
            // 右侧点判断
            if(polygon[i][0]>border_points[2][0]){
                border_points[2][0]=polygon[i][0];
                border_points[2][1]=polygon[i][1];
            }
            // 上侧点判断
            if(polygon[i][1]>border_points[3][1]){
                border_points[3][0]=polygon[i][0];
                border_points[3][1]=polygon[i][1];
            }
        };
    };
    
    // 获得当前排样的宽度
    static double getPolysRight(vector<Polygon> all_polys){
        double length=0;
        for(Polygon poly:all_polys){
            vector<double> pt;
            getRightPt(poly,pt);
            if(pt[0]>length){
                length=pt[0];
            }
        }
        return length;
    };
    // 获得全部的边（从头到尾）
    static void getPolyEdges(Polygon poly, vector<Polygon> &all_edges){
        int pt_num = (int)poly.size();
        for(int i = 0; i < pt_num - 1; i++){
            all_edges.push_back({poly[i],poly[i+1]});
        }
        if(poly[0][0] != poly[pt_num-1][0] || poly[0][1] != poly[pt_num-1][1]){
            all_edges.push_back({poly[pt_num-1],poly[0]});
        }
    }
};

