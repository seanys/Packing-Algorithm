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

using namespace boost::geometry;
using namespace std;

#define BIAS 0.000001

//主要包含注册多边形、转化多边形
class GeometryProcess{
    // 数组转化为多边形
    static void convertPoly(){
        // 定义各种参数
        typedef model::d2::point_xy<double> point_type;
        point_type a;
        model::linestring<point_type> b;
        model::polygon<point_type> poly;
        // 读取参数并输入
        read_wkt("POINT(1 2)", a);
        read_wkt("LINESTRING(0 0,2 2,3 1)", b);
        read_wkt("POLYGON((0 0,0 7,4 2,2 0,0 0))", poly);
        // 输出wkt模式的结果
        cout << wkt(poly) << endl;
    }
    
};

// 单个多边形处理
class PolygonFunctions{
public:
    // 获得多边形的边界
    static void polyBound(){
        polyBoundPoints(); // 获得边界点
        
    };
    // 获得多边形的边界点
    static void polyBoundPoints(){
        typedef model::d2::point_xy<double> point;
        model::polygon<point> poly;
        read_wkt("POLYGON((0 0,0 4,4 0,0 0))", poly);
//        for_each_point(poly, list_coordinates<point>);
    };
    // 计算多边形的重心
    static void polyCentroid(){
        typedef model::d2::point_xy<double> point_type;
        typedef model::polygon<point_type> polygon_type;

        polygon_type poly;
        read_wkt(
            "POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)"
                "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))", poly);

        point_type p;
        centroid(poly, p);

        cout << "centroid: " << dsv(p) << std::endl;

    };
    // 判读那多边形是否为空
    static bool isEmpty(){
        model::multi_linestring
            <
                model::linestring
                    <
                        model::d2::point_xy<double>
                    >
            > mls;
        read_wkt("MULTILINESTRING((0 0,0 10,10 0),(1 1,8 1,1 8))", mls);
        cout << "Is empty? " << (boost::geometry::is_empty(mls) ? "yes" : "no") << endl;
        clear(mls);
        cout << "Is empty (after clearing)? " << (boost::geometry::is_empty(mls) ? "yes" : "no") << endl;
        return true;
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
        typedef boost::geometry::model::d2::point_xy<double> P;
        boost::geometry::model::linestring<P> line1, line2;

        read_wkt("linestring(1 1,2 2,3 3)", line1);
        read_wkt("linestring(2 1,1 2,4 0)", line2);

        bool b = boost::geometry::intersects(line1, line2);

        std::cout << "Intersects: " << (b ? "YES" : "NO") << std::endl;
    }
    // 计算多边形的差集
    void polysIntersection(){
        typedef model::polygon<model::d2::point_xy<double> > polygon;

        polygon green, blue;

        read_wkt(
            "POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)"
                "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))", green);

        read_wkt(
            "POLYGON((4.0 -0.5 , 3.5 1.0 , 2.0 1.5 , 3.5 2.0 , 4.0 3.5 , 4.5 2.0 , 6.0 1.5 , 4.5 1.0 , 4.0 -0.5))", blue);

        deque<polygon> output;
        intersection(green, blue, output);

        int i = 0;
        cout << "green && blue:" << endl;
        BOOST_FOREACH(polygon const& p, output)
        {
            cout << i++ << ": " << area(p) << endl;
        }

    }
    // 计算多边形的交集
    void polysUnion(){
        typedef model::polygon<model::d2::point_xy<double> > polygon;

        polygon green, blue;

        read_wkt("POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)"
                   "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))", green);

        read_wkt(
               "POLYGON((4.0 -0.5 , 3.5 1.0 , 2.0 1.5 , 3.5 2.0 , 4.0 3.5 , 4.5 2.0 , 6.0 1.5 , 4.5 1.0 , 4.0 -0.5))", blue);

        vector<polygon> output;
        union_(green, blue, output);

        int i = 0;
        cout << "green || blue:" << endl;
        BOOST_FOREACH(polygon const& p, output)
        {
            cout << i++ << ": " << area(p) << endl;
        }
    }
    // 计算多边形的差集合
    void polysDifference(){
        typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > polygon;

        polygon green, blue;

        boost::geometry::read_wkt(
            "POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)"
                "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))", green);

        boost::geometry::read_wkt(
            "POLYGON((4.0 -0.5 , 3.5 1.0 , 2.0 1.5 , 3.5 2.0 , 4.0 3.5 , 4.5 2.0 , 6.0 1.5 , 4.5 1.0 , 4.0 -0.5))", blue);

        std::list<polygon> output;
        boost::geometry::difference(green, blue, output);

        int i = 0;
        std::cout << "green - blue:" << std::endl;
        BOOST_FOREACH(polygon const& p, output)
        {
            std::cout << i++ << ": " << boost::geometry::area(p) << std::endl;
        }


        output.clear();
        boost::geometry::difference(blue, green, output);

        i = 0;
        std::cout << "blue - green:" << std::endl;
        BOOST_FOREACH(polygon const& p, output)
        {
            std::cout << i++ << ": " << boost::geometry::area(p) << std::endl;
        }

    }
};
