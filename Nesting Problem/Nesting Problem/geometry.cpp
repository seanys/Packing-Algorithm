//
//  geometry.cpp
//  Nesting Problem
//
//  Created by 爱学习的兔子 on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//

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


//主要包含注册多边形、转化多边形
class GeometryProcess{
public:
    /*
     数组转化为多边形
     */
    static void convertPoly(vector<vector<double>> poly, Polygon Poly){
        // 首先全部转化为wkt格式
        string wkt_poly="POLYGON((";
        for (int i = 0; i < poly.size();i++)
        {
            wkt_poly+=to_string(poly[i][0]) + " " + to_string(poly[i][0])+",";
        };
        wkt_poly+=to_string(poly[0][0]) + " " + to_string(poly[0][0])+"))";
        // 然后读取到Poly中
        read_wkt(wkt_poly, Poly);
    }
    
};

class PackingAssistant{
public:
    static void getIFR(){
        
    }
};





// 几何对象遍历的函数
class GeometryFunctions{
private:
    string _type="slide";
public:
    PolygonFunctions(string opertor_type){
        _type=opertor_type;
    };
    inline void operator()(Point& p){
        if(_type=="slide"){
            
        }
    };
    
};

// 单个多边形处理
class PolygonAssign{
public:
    /*
     获得多边形的边界（x_min,x_max,y_min,y_max）
     */
    static void polyBound(){
        polyBoundPoints(); // 获得边界点
        
    };
    /*
     平移多边形到目标位置（直接处理对象）
     */
    static void slidePoly(){
        Point point;

        boost::geometry::set<0>(point, 1);
        boost::geometry::set<1>(point, 2);

        cout << "Location: " << dsv(point) << endl;
    };
    /*
     遍历全部的点
     */
    template <typename Point>
    static void traverseTest(Point const& pt){
        cout<<get<0>(pt) <<","<<get<0>(pt) <<endl;
    };
    /*
     获得边界的全部点
     */
    static void polyBoundPoints(){
        Polygon poly;
        read_wkt("POLYGON((0 0,0 4,4 0,0 0))", poly);
//        cout<<dsv(get<0>(poly))<<endl;
        for_each_point(poly, traverseTest<Point>);
    };
    /*
     获得多边形的重心
     */
    static void polyCentroid(){
        Polygon poly;
        read_wkt(
            "POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)"
                "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))", poly);

        Point p;
        centroid(poly, p);
        
        // 输出dsv格式，也就是数组格式
        cout << "centroid: " << dsv(p) << endl;

    };
    /*
     判断是否为空
     */
    static bool isEmpty(){
        model::multi_linestring <LineString> mls;
        read_wkt("MULTILINESTRING((0 0,0 10,10 0),(1 1,8 1,1 8))", mls);
        cout << "Is empty? " << (boost::geometry::is_empty(mls) ? "yes" : "no") << endl;
        clear(mls);
        cout << "Is empty (after clearing)? " << (boost::geometry::is_empty(mls) ? "yes" : "no") << endl;
        return true;
    };
    /*
     增加形状
     */
    static void addPoint(){
        LineString line;
        line.push_back(Point(0, 0));
        line.push_back(Point(1, 1));

//        cout<< dsv(line | boost::adaptors::reversed) << endl;
    };
    /*
     注册一个有洞的多边形，后续带Start Point的需要计算
     */
    static void assignPolyWithHole(){
//        typedef polygon::polygon_with_holes_data<int> polygon;
//        typedef polygon::polygon_traits<polygon>::point_type point;
//
//        typedef boost::polygon::polygon_with_holes_traits<polygon>::hole_type hole;
//
//        point pts[5] = {
//            boost::polygon::construct<Point>(0, 0),
//            boost::polygon::construct<Point>(0, 10),
//            boost::polygon::construct<Point>(10, 10),
//            boost::polygon::construct<Point>(10, 0),
//            boost::polygon::construct<Point>(0, 0)
//        };
//        point hole_pts[5] = {
//            boost::polygon::construct<point>(1, 1),
//            boost::polygon::construct<point>(9, 1),
//            boost::polygon::construct<point>(9, 9),
//            boost::polygon::construct<point>(1, 9),
//            boost::polygon::construct<point>(1, 1)
//        };
//
//        hole hls[1];
//        boost::polygon::set_points(hls[0], hole_pts, hole_pts+5);
//
//        polygon poly;
//        boost::polygon::set_points(poly, pts, pts+5);
//        boost::polygon::set_holes(poly, hls, hls+1);
//
//        std::cout << "Area (using Boost.Geometry): "
//            << boost::geometry::area(poly) << std::endl;
//        std::cout << "Area (using Boost.Polygon): "
//            << boost::polygon::area(poly) << std::endl;
    }
};


// 计算多个多边形的关系
class PolygonsAssistant{
public:
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
    // 计算多边形的差集合
    void polysDifference(){
        // 测试基础
        Polygon green, blue;

        list<Polygon> output;
        difference(green, blue, output);

        int i = 0;
        cout << "green - blue:" << endl;
        BOOST_FOREACH(Polygon const& p, output)
        {
            std::cout << i++ << ": " << area(p) << std::endl;
        }

        output.clear();
        difference(blue, green, output);

        i = 0;
        cout << "blue - green:" << endl;
        BOOST_FOREACH(Polygon const& p, output)
        {
            std::cout << i++ << ": " << boost::geometry::area(p) << std::endl;
        }

    }
};
