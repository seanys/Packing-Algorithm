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

class GeoFunc{
private:
    int test=1;
public:
    int getIntersection(){
        model::d2::point_xy<int> p1(1, 1), p2(2, 2);
        cout << "Distance p1-p2 is: " << distance(p1, p2) << endl;
        return 0;
    }
    // 判断多边形是否香蕉
    void polysIntersects(){
        typedef boost::geometry::model::d2::point_xy<double> P;
        boost::geometry::model::linestring<P> line1, line2;

        boost::geometry::read_wkt("linestring(1 1,2 2,3 3)", line1);
        boost::geometry::read_wkt("linestring(2 1,1 2,4 0)", line2);

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
    // 判读那多边形是否为空
    bool isEmpty(){
        model::multi_linestring
            <
                model::linestring
                    <
                        model::d2::point_xy<double>
                    >
            > mls;
        read_wkt("MULTILINESTRING((0 0,0 10,10 0),(1 1,8 1,1 8))", mls);
        cout << "Is empty? " << (boost::geometry::is_empty(mls) ? "yes" : "no") << std::endl;
        boost::geometry::clear(mls);
        std::cout << "Is empty (after clearing)? " << (boost::geometry::is_empty(mls) ? "yes" : "no") << std::endl;
        return true;
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
    // 计算多边形的重心
    void polyCentroid(){
        typedef boost::geometry::model::d2::point_xy<double> point_type;
        typedef boost::geometry::model::polygon<point_type> polygon_type;

        polygon_type poly;
        boost::geometry::read_wkt(
            "POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)"
                "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))", poly);

        point_type p;
        boost::geometry::centroid(poly, p);

        std::cout << "centroid: " << boost::geometry::dsv(p) << std::endl;

    }
};
