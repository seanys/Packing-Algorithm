//
//  geometry.cpp
//  Nesting Problem
//
//  Created by 爱学习的兔子 on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//

#include "data_assistant.cpp"
#include <deque>
#include <iostream>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/foreach.hpp>
#include <boost/geometry/algorithms/for_each.hpp>

using namespace boost::geometry;
using namespace std;

#define BIAS 0.000001

// 基础定义
typedef model::d2::point_xy<double> Point;
typedef model::polygon<Point> Polygon;
typedef model::linestring<Point> LineString;

//read_wkt(
//    "POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)"
//        "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))", green);

//read_wkt(
//    "POLYGON((4.0 -0.5 , 3.5 1.0 , 2.0 1.5 , 3.5 2.0 , 4.0 3.5 , 4.5 2.0 , 6.0 1.5 , 4.5 1.0 , 4.0 -0.5))", blue);

// 包含处理函数
class PackingAssistant{
public:
    /*
     获得Inner Fit Rectangle
     */
    static void getIFR(VectorPoints polygon,double container_width,double container_length,VectorPoints &IFR){
        // 初始参数，获得多边形特征
        VectorPoints border_points;
        getBorder(polygon,border_points);
        double poly_width_left=border_points[3][0]-border_points[0][0];
        double poly_width_right=border_points[2][0]-border_points[3][0];
        double poly_height=border_points[3][1]-border_points[1][1];

        // IFR具体计算（从左上角逆时针计算）
        IFR.push_back({poly_width_left,container_width});
        IFR.push_back({poly_width_left,poly_height});
        IFR.push_back({container_length-poly_width_right,poly_height});
        IFR.push_back({container_length-poly_width_right,container_width});
    };
    /*
     移动某个多边形
     */
    static void slidePoly(VectorPoints &polygon,double delta_x,double delta_y){
        for(int i=0;i<polygon.size();i++){
            polygon[i][0]=polygon[i][0]+delta_x;
            polygon[i][1]=polygon[i][1]+delta_y;
        }
    };
    /*
     移动多边形到某个位置（参考点）
     */
    static void slideToPosition(VectorPoints &polygon,vector<double> target_pt){
        vector<double> refer_pt;
        getReferPt(polygon,refer_pt);
        double delta_x=target_pt[0]-refer_pt[0];
        double delta_y=target_pt[1]-refer_pt[1];
        for(int i=0;i<polygon.size();i++){
            polygon[i][0]=polygon[i][0]+delta_x;
            polygon[i][1]=polygon[i][1]+delta_y;
        }
    };
    /*
     获得多边形的所有的边界情况min_x min_y max_x max_y
     */
    static void getBound(VectorPoints polygon,vector<double> &bound){
        VectorPoints border_points;
        getBorder(polygon,border_points);
        bound={border_points[0][0],border_points[1][1],border_points[2][0],border_points[3][1]};
    };
    /*
     遍历获得一个多边形的最左侧点
     */
    static void getBottomLeft(VectorPoints polygon,vector<double> &bl_point){
        bl_point={999999999,999999999};
        for(auto point:polygon){
            if(point[0]<bl_point[0] || (point[0]==bl_point[0]&&point[1]<bl_point[1]) ){
                bl_point[0]=point[0];
                bl_point[1]=point[1];
            }
        };
    };
    /*
     仅仅获得参考点，是第一个Top位置
     */
    static void getReferPt(VectorPoints polygon,vector<double> &refer_pt){
        refer_pt={0,-9999999999};
        for(int i=0;i<polygon.size();i++){
            if(polygon[i][1]>refer_pt[1]){
                refer_pt[0]=polygon[i][0];
                refer_pt[1]=polygon[i][1];
            }
        }
    };
    /*
     仅仅获得底部位置（用于NFP计算），是第一个Bottom位置
     */
    static void getBottomPt(VectorPoints polygon,vector<double> &bottom_pt){
        bottom_pt={0,9999999999};
        for(int i=0;i<polygon.size();i++){
            if(polygon[i][1]<bottom_pt[1]){
                bottom_pt[0]=polygon[i][0];
                bottom_pt[1]=polygon[i][1];
            }
        }
    };
    /*
     获得多边形的边界四个点，border_points有left bottom right top四个点
     暂时不考虑参考点，参考点统一逆时针旋转第一个最上方的点
     */
    static void getBorder(VectorPoints polygon,VectorPoints &border_points){
        // 增加边界的几个点
        border_points.push_back(vector<double>{9999999999,0});
        border_points.push_back(vector<double>{0,999999999});
        border_points.push_back(vector<double>{-999999999,0});
        border_points.push_back(vector<double>{0,-999999999});
        // 遍历所有的点，分别判断是否超出界限
        for(auto point:polygon){
            // 左侧点判断
            if(point[0]<border_points[0][0]){
                border_points[0][0]=point[0];
                border_points[0][1]=point[1];
            }
            // 下侧点判断
            if(point[1]<border_points[1][1]){
                border_points[1][0]=point[0];
                border_points[1][1]=point[1];
            }
            // 右侧点判断
            if(point[0]>border_points[2][0]){
                border_points[2][0]=point[0];
                border_points[2][1]=point[1];
            }
            // 上侧点判断
            if(point[1]>border_points[3][1]){
                border_points[3][0]=point[0];
                border_points[3][1]=point[1];
            }
        };
    };
};

// 获得NFP
class NFPAssistant{
protected:
    csv::Reader nfp_result;
    int poly_num;
    int orientation_num;
    vector<VectorPoints> NPFs; // 存储全部的NFP，按行存储
public:
    /*
     预加载全部的NFP，直接转化到NFP中
     */
    NFPAssistant(string _path,int poly_num,int orientation_num){
//        nfp_result.read(_path);
        nfp_result.read("/Users/sean/Documents/Projects/Data/fu.csv");
        this->poly_num=poly_num;
        this->orientation_num=orientation_num;
        cout<<"加载全部NFP"<<endl;
        while(nfp_result.busy()) {
            if (nfp_result.ready()) {
                auto row = nfp_result.next_row();
                VectorPoints nfp;
                if(row["nfp"]!=""){
                    DataAssistant::load2DVector(row["nfp"],nfp);
                    NPFs.push_back(nfp);
                }
            }
        }
    };
    /*
     读取NFP的确定行数，i为固定形状，j为非固定形状,oi/oj为形状
     */
    void getNFP(int i,int j, int oi, int oj, VectorPoints poly_j ,VectorPoints &nfp){
        // 获得原始的NFP
        int row_num= i*192+j*16+oi*4+oj;
        nfp=NPFs[row_num];
        // 将NFP移到目标位置
        vector<double> bottom_pt;
        PackingAssistant::getBottomPt(poly_j,bottom_pt);
        PackingAssistant::slidePoly(nfp,bottom_pt[0],bottom_pt[1]);
    }
};

// 封装获得全部点的函数
template <typename Point>
class AllPoint{
private :
    VectorPoints *temp_all_points;
public :
    AllPoint(VectorPoints *all_points){
        temp_all_points=all_points;
    };
    inline void operator()(Point& pt)
    {
        vector<double> new_pt={get<0>(pt),get<1>(pt)};
        (*temp_all_points).push_back(new_pt);
    }
};

//主要包含注册多边形、转化多边形
class GeometryProcess{
public:
    /*
     数组转化为多边形
     */
    static void convertPoly(vector<vector<double>> poly, Polygon &Poly){
        // 首先全部转化为wkt格式
        string wkt_poly="POLYGON((";
        for (int i = 0; i < poly.size();i++)
        {
            wkt_poly+=to_string(poly[i][0]) + " " + to_string(poly[i][1])+",";
        };
        wkt_poly+=to_string(poly[0][0]) + " " + to_string(poly[0][1])+"))";
        // 然后读取到Poly中
        read_wkt(wkt_poly, Poly);
    };
    /*
     通过for each point遍历
     */
    static void getAllPoints(list<Polygon> all_polys,VectorPoints &all_points){
        for(auto poly:all_polys){
            VectorPoints temp_points;
            getGemotryPoints(poly,temp_points);
            all_points.insert(all_points.end(),temp_points.begin(),temp_points.end());
        }
    };
    static void getGemotryPoints(Polygon poly,VectorPoints &temp_points){
        for_each_point(poly, AllPoint<Point>(&temp_points));
    }
};

// 处理多个多边形的关系
class PolygonsOperator{
public:
    // 计算多边形的差集合
    static void polysDifference(list<Polygon> &feasible_region, Polygon nfp_poly){
        // 逐一遍历求解重叠
        list<Polygon> new_feasible_region;
        for(auto region_item:feasible_region){
            // 计算差集
            list<Polygon> output;
            cout<<endl<<"region_item:"<<wkt(region_item)<<endl;
            cout<<"nfp_poly:"<<wkt(nfp_poly)<<endl<<endl;
            cout<<endl<<"region_item:"<<dsv(region_item)<<endl;
            cout<<"nfp_poly:"<<dsv(nfp_poly)<<endl<<endl;

            intersection(region_item, nfp_poly, output);
            DataAssistant::appendList(new_feasible_region,output);
            for(auto item:output){
                cout<<"output:"<<dsv(item)<<endl;
            };
        };
        // 将新的Output全部输入进去
        feasible_region.clear();
        copy(new_feasible_region.begin(), new_feasible_region.end(), back_inserter(feasible_region));
    }
    // 获得两条直线的交点
    int getIntersection(){
        model::d2::point_xy<int> p1(1, 1), p2(2, 2);
        cout << "Distance p1-p2 is: " << distance(p1, p2) << endl;
        return 0;
    }
    // 判断多边形是否相交
    void polysIntersects(){
        LineString line1, line2;

        read_wkt("linestring(1 1,2 2,3 3)", line1);
        read_wkt("linestring(2 1,1 2,4 0)", line2);

        bool b = intersects(line1, line2);

        cout << "Intersects: " << (b ? "YES" : "NO") << endl;
    }
    // 计算多边形的差集
    void polysIntersection(){
        // 测试基础
        Polygon green, blue;

        deque<Polygon> output;
        intersection(green, blue, output);

        int i = 0;
        cout << "green && blue:" << endl;
        BOOST_FOREACH(Polygon const& p, output)
        {
            cout << i++ << ": " << area(p) << endl;
        }

    }
    // 计算多边形的交集
    void polysUnion(){
        // 测试基础
        Polygon green, blue;

        vector<Polygon> output;
        union_(green, blue, output);

        int i = 0;
        cout << "green || blue:" << endl;
        BOOST_FOREACH(Polygon const& p, output)
        {
            cout << i++ << ": " << area(p) << endl;
        }
    }
};
