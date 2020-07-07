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
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/foreach.hpp>
#include <boost/geometry/algorithms/for_each.hpp>
#include <algorithm>
#include <fstream>
#include <time.h>

#define BIAS 0.000001

using namespace std;
using namespace x2struct;
using namespace boost::geometry;

typedef vector<vector<double>> Polygon;

typedef model::d2::point_xy<double> PointBoost;
typedef model::polygon<PointBoost> PolygonBoost;
typedef model::linestring<PointBoost> LineStringBoost;

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
    // 保留小数的计算
    static void round(double &value, int precison){
        double m = pow(10, precison);
        value = floor(value * m + 0.5)/m;
    };
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
    // 获得字符串时间
    static string getTimeString(){
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer,sizeof(buffer),"%m-%d %H:%M:%S",timeinfo);
        string str(buffer);
        return str;
    };
    template <typename T>
    static string vector1DToString(vector<T> &vec){
        string str = "[";
        for (int i = 0; i < vec.size(); i++){
            str += to_string(vec[i]);
            if(i < (int)vec.size() - 1){
                str += ",";
            }
        }
        str += "]";
        return str;
    };
    template <typename T>
    static string vector2DToString(vector<vector<T>> &vec){
        string str = "[", temp_str;
        for (int i = 0; i < vec.size(); i++){
            string temp_str = vector1DToString(vec[i]);
            str += temp_str;
            if(i < (int)vec.size() - 1){
                str += ",";
            }
        }
        str += "]";
        return str;
    };
    template <typename T>
    static string vector3DToString(vector<vector<vector<T>>> &vec){
        string str = "[", temp_str;
        for (int i = 0; i < vec.size(); i++){
            string temp_str = vector2DToString(vec[i]);
            str += temp_str;
            if(i < (int)vec.size() - 1){
                str += ",";
            }
        }
        str += "]";
        return str;
    }
};

/*
 CSV的读写于记录
 */
class CSVAssistant{
public:
    // 将形状输出到目标的文件夹
    static void recordPolys(string _path,vector<Polygon> all_polys){
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
    // 记录一下容器的宽度
    static void recordContainer(string _path, double width, double length){
        // 读取并设置文件头
        csv::Writer foo(_path);
        foo.configure_dialect()
        .delimiter(", ")
        .column_names("container");
        
        foo.write_row(to_string(width));
        foo.write_row(to_string(length));
        foo.close();
    };
    // 记录成功的情况
    static void recordSuccess(string set_name, double length, double ratio, vector<double> orientation, vector<Polygon> polys){
        ofstream ofs;
        string root = "/Users/sean/Documents/Projects/Packing-Algorithm/record/c_plt/";
        string path = root + set_name + ".csv";
        ofs.open (path, ofstream::out | ofstream::app);

        ofs << TOOLS::getTimeString() << "," << length << "," << ratio << ",\"" << TOOLS::vector1DToString(orientation) << "\",\"" << TOOLS::vector3DToString(polys) << "\"" << endl;
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
    static void showPolys(vector<Polygon> all_polys, double width, double length){
        string polys_path = "/Users/sean/Documents/Projects/Packing-Algorithm/record/c_plt/c_record.csv";
        CSVAssistant::recordPolys(polys_path,all_polys);
        string container_path = "/Users/sean/Documents/Projects/Packing-Algorithm/record/c_plt/container.csv";
        CSVAssistant::recordContainer(container_path, width, length);
        system("/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7 /Users/sean/Documents/Projects/Packing-Algorithm/tools/cplt.py");
    };
    // 单个形状的加载
    static void polyShow(Polygon poly){
        vector<Polygon> all_polys = {poly};
//        polysShow(all_polys);
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
    // 获得垂直点
    static void getFootPoint(vector<double> point, Polygon edge, vector<double> &foot_pt){
        double x0 = point[0];
        double y0 = point[1];
        
        double x1 = edge[0][0];
        double y1 = edge[0][1];
        
        double x2 = edge[1][0];
        double y2 = edge[1][1];
        
        double k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * 1.0;
        
        foot_pt = {k * (x2 - x1) + x1, k * (y2 - y1) + y1};
    };
    // 获得Inner Fit Rectangle
    static void getIFR(Polygon poly,double width,double length,Polygon &IFR){
        // 初始参数，获得多边形特征
        Polygon border_points;
        getBorder(poly,border_points);
        
        double poly_width_left = border_points[3][0] - border_points[0][0];
        double poly_width_right = border_points[2][0] - border_points[3][0];
        double poly_height = border_points[3][1] - border_points[1][1];

        // IFR具体计算（从左上角顺时针计算）
        IFR.push_back({poly_width_left, width});
        IFR.push_back({poly_width_left, poly_height});
        IFR.push_back({length - poly_width_right,poly_height});
        IFR.push_back({length - poly_width_right,width});
    };
    // 移动某个多边形
    static void slidePoly(Polygon &polygon,double delta_x,double delta_y){
        for(int i=0;i<polygon.size();i++){
            polygon[i][0] = polygon[i][0]+delta_x;
            polygon[i][1] = polygon[i][1]+delta_y;
        }
    };
    // 移动多边形到某个位置（参考点）
    static void slideToPoint(Polygon &polygon,vector<double> target_pt){
        vector<double> refer_pt;
        getTopPt(polygon,refer_pt);
        double delta_x = target_pt[0] - refer_pt[0];
        double delta_y = target_pt[1] - refer_pt[1];
        for(int i = 0; i < polygon.size(); i ++){
            polygon[i][0] = polygon[i][0] + delta_x;
            polygon[i][1] = polygon[i][1] + delta_y;
        }
    };
    // 获得多边形的所有的边界情况min_x min_y max_x max_y
    static void getBound(Polygon polygon,vector<double> &bound){
        Polygon border_points;
        getBorder(polygon,border_points);
        bound = {border_points[0][0],border_points[1][1],border_points[2][0],border_points[3][1]};
    };
    // 仅仅获得最右侧点，同样为逆时针处理（用于判断是逗超出界限）
    static void getRightPt(Polygon polygon,vector<double> &right_pt){
        right_pt = {-9999999999,0};
        int poly_size = (int)polygon.size();
        for(int i = 0; i < poly_size; i ++){
            if(polygon[i][0] > right_pt[0]){
                right_pt[0] = polygon[i][0];
                right_pt[1] = polygon[i][1];
            }
        }
    };
    // 获得多边形参考点
    static void getTopPt(Polygon polygon,vector<double> &top_pt){
        top_pt = {0,-9999999999};
        int poly_size = (int)polygon.size();
        for(int i = 0; i < poly_size; i ++){
            if(polygon[i][1] > top_pt[1]){
                top_pt[0] = polygon[i][0];
                top_pt[1] = polygon[i][1];
            }
        }
    };
    // 获得多边形的底部的点
    static void getBottomPt(Polygon polygon,vector<double> &bottom_pt){
        bottom_pt = {0,9999999999};
        int poly_size = (int)polygon.size();
        for(int i = 0; i < poly_size; i ++){
            if(polygon[i][1] < bottom_pt[1]){
                bottom_pt[0] = polygon[i][0];
                bottom_pt[1] = polygon[i][1];
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
        int poly_size = (int)polygon.size();
        for(int i = 0; i < poly_size; i ++){
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
            if(pt[0] > length){
                length = pt[0];
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

// 封装获得全部点的函数
template <typename Point>
class AllPoint{
private :
    Polygon *temp_all_points;
public :
    AllPoint(Polygon *all_points){
        temp_all_points=all_points;
    };
    inline void operator()(Point& pt)
    {
        vector<double> new_pt={get<0>(pt),get<1>(pt)};
        (*temp_all_points).push_back(new_pt);
    }
};

//主要包含注册多边形、转化多边形
class GeometryAssistant{
public:
    // 判断多边形是否包含点（注意需要考虑）
    static bool containPoint(Polygon poly, vector<double> pt){
        PolygonBoost Poly;
        convertPoly(poly, Poly); // 转化为多边形
        PointBoost PT(pt[0],pt[1]); // 处理点的对象
        return  within(PT, Poly);
    };
    // 将形状换一个方向才，再进一步处理
    static void reversePolygon(Polygon &poly){
        reverse(poly.begin(),poly.end());
    };
    // 切割IFR计算差集
    static void cutIFR(Polygon ifr, vector<Polygon> nfps, vector<vector<double>> &feasible_points){
        PolygonBoost IFR; convertPoly(ifr, IFR);
        list<PolygonBoost> all_feasible_regions = {IFR}, temp_feasible_region;
        for(auto nfp: nfps){
            PolygonBoost NFP; convertPoly(nfp, NFP); temp_feasible_region.clear();
            for(auto feasible_region: all_feasible_regions){
                list<PolygonBoost> output;
                difference(feasible_region, NFP, output);
                appendList(temp_feasible_region, output);
            }
            all_feasible_regions = temp_feasible_region;
        }
        if((int)all_feasible_regions.size()==0){
            feasible_points = {};
            return;
        }
        vector<Polygon> feasible_polys;
        boostListToVector(all_feasible_regions, feasible_polys); // 转存到可行的多边形
        for(auto poly:feasible_polys){
            feasible_points.insert(feasible_points.end(),poly.begin(),poly.end());
        }
    };
    // 计算IFR/NFP1/NFP2的交集
    static void getNFPInter(Polygon nfp1, Polygon nfp2, Polygon ifr, vector<vector<double>> &inter_points){
        PolygonBoost NFP1, NFP2, IFR;
        convertPoly(nfp1, NFP1); convertPoly(nfp2, NFP2); convertPoly(ifr, IFR);
        list<PolygonBoost> nfp_inter, temp_inter, ifr_inter;
        intersection(NFP1, NFP2, nfp_inter);
        // 对交集逐一与IFR计算交点
        for(auto GeoItem: nfp_inter){
            intersection(IFR, GeoItem, temp_inter);
            appendList(ifr_inter, temp_inter);
        }
        // 遍历全部的相交区域
        for(auto GeoItem: ifr_inter){
            vector<vector<double>> temp_pts;
            boostToVector(GeoItem, temp_pts);
            if((int)temp_pts.size() == 2) continue; // 如果是直线则不处理
            inter_points.insert(inter_points.end(), temp_pts.begin(), temp_pts.end());
        }
    };
    // 计算IFR/NFP1/NFP2的交集
    static void getNFPIFRInter(Polygon nfp, Polygon ifr, vector<vector<double>> &inter_points){
        PolygonBoost NFP, IFR;
        convertPoly(nfp, NFP); convertPoly(ifr, IFR);
        list<PolygonBoost> ifr_inter, temp_inter;
        intersection(NFP, IFR, ifr_inter);
        // 遍历全部的相交区域
        for(auto GeoItem: ifr_inter){
            vector<vector<double>> temp_pts;
            boostToVector(GeoItem, temp_pts);
            if((int)temp_pts.size() == 2) continue; // 如果是直线则不处理
            inter_points.insert(inter_points.end(), temp_pts.begin(), temp_pts.end());
        }
    };
    // 数组转化为多边形
    static void convertPoly(Polygon poly, PolygonBoost &Poly){
        reversePolygon(poly);
        // 空集的情况
        if(poly.size() == 0){
            read_wkt("POLYGON(())", Poly);
            return;
        }
        // 首先全部转化为wkt格式
        string wkt_poly = "POLYGON((";
        for (int i = 0; i < poly.size();i++){
            wkt_poly += to_string(poly[i][0]) + " " + to_string(poly[i][1]) + ",";
            if(i == poly.size()-1){
                wkt_poly += to_string(poly[0][0]) + " " + to_string(poly[0][1]) + "))";
            }
        };
        // 然后读取到Poly中
        read_wkt(wkt_poly, Poly);
    };
    // 获得某个集合对象的全部点
    static void boostToVector(PolygonBoost poly,Polygon &temp_points){
        for_each_point(poly, AllPoint<PointBoost>(&temp_points));
    };
    // 将List Boost对象转化为
    static void boostListToVector(list<PolygonBoost> Polys,vector<Polygon> &all_polys){
        for(PolygonBoost poly_item: Polys){
            Polygon poly_points;
            boostToVector(poly_item,poly_points);
            all_polys.push_back(poly_points);
        }
    };
    // 获得多边形的全部点
    static void getAllPoints(list<PolygonBoost> all_polys,Polygon &all_points){
        for(auto poly_item:all_polys){
            Polygon temp_points;
            boostToVector(poly_item,temp_points);
            all_points.insert(all_points.end(),temp_points.begin(),temp_points.end());
        }
    };
    // 增加List链接
    template <typename T>
    static void appendList(list<T> &old_list,list<T> &new_list){
        for(auto item:new_list){
            old_list.push_back(item);
        }
    };
};
