//
//  boost_geo.cpp
//  Simple LP Search
//
//  Created by 爱学习的兔子 on 2020/7/6.
//  Copyright © 2020 Yang Shan. All rights reserved.
//

#include <vector>
#include <iostream>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/foreach.hpp>
#include <boost/geometry/algorithms/for_each.hpp>

using namespace boost::geometry;
using namespace std;

typedef model::d2::point_xy<double> PointBoost;
typedef model::polygon<PointBoost> PolygonBoost;
typedef model::linestring<PointBoost> LineStringBoost;

typedef vector<vector<double>> Polygon;

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
class GeometryProcess{
public:
    // 将形状换一个方向才，再进一步处理
    static void reversePolygon(Polygon poly){
        
    };
    // 切割IFR计算差集
    static void cutIFR(Polygon ifr, vector<Polygon> nfps){
        
    };
    // 计算IFR/NFP1/NFP2的交集
    static void getPolysInter(vector<Polygon> polys){
        
    };
    
    // 数组转化为多边形
    static void convertPoly(vector<vector<double>> poly, PolygonBoost &Poly){
        // 空集的情况
        if(poly.size()==0){
            read_wkt("POLYGON(())", Poly);
            return;
        }
        // 首先全部转化为wkt格式
        string wkt_poly="POLYGON((";
        for (int i = 0; i < poly.size();i++){
            wkt_poly+=to_string(poly[i][0]) + " " + to_string(poly[i][1]) + ",";
            if(i==poly.size()-1){
                wkt_poly+=to_string(poly[0][0]) + " " + to_string(poly[0][1]) + "))";
            }
        };
        // 然后读取到Poly中
        read_wkt(wkt_poly, Poly);
    };
    // 获得多边形的全部点
    static void getAllPoints(list<PolygonBoost> all_polys,Polygon &all_points){
        for(auto poly_item:all_polys){
            Polygon temp_points;
            getGemotryPoints(poly_item,temp_points);
            all_points.insert(all_points.end(),temp_points.begin(),temp_points.end());
        }
    };
    // 获得vector<list<VectorPoints>>的多边形（并非全部点）
    static void getListPolys(vector<list<PolygonBoost>> list_polys,vector<Polygon> &all_polys){
        for(auto _list:list_polys){
            for(PolygonBoost poly_item:_list){
                Polygon poly_points;
                getGemotryPoints(poly_item,poly_points);
                all_polys.push_back(poly_points);
            }
        }
    };
    // 获得某个集合对象的全部点
    static void getGemotryPoints(PolygonBoost poly,Polygon &temp_points){
        for_each_point(poly, AllPoint<PointBoost>(&temp_points));
    };
};
