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
#include <algorithm>

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
class GeometryAssistant{
public:
    // 判断多边形是否包含点（注意需要考虑）
    static bool containPoint(Polygon poly, vector<double> pt){
        reversePolygon(poly); // 逆向处理
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
    static void cutIFR(Polygon ifr, vector<Polygon> nfps){
        
    };
    // 计算IFR/NFP1/NFP2的交集
    static void getPolysInter(vector<Polygon> polys){
        
    };
    
//    // 计算多边形的差集合
//    static void polysDifference(list<Polygon> &feasible_region, Polygon sub_region){
//        // 逐一遍历求解重叠
//        list<Polygon> new_feasible_region;
//        for(auto region_item:feasible_region){
//            list<Polygon> output;
//            difference(region_item, sub_region, output);
//            appendList(new_feasible_region,output);
//        };
//        // 将新的Output全部输入进去
//        feasible_region.clear();
//        copy(new_feasible_region.begin(), new_feasible_region.end(), back_inserter(feasible_region));
//    }
//    // 逐一遍历求差集
//    static void polyListDifference(list<Polygon> &feasible_region, list<Polygon> sub_region){
//        for(auto region_item:sub_region){
//            polysDifference(feasible_region,region_item);
//        }
//    }
//    // List和一个Poly的差集
//    static void listToPolyIntersection(list<Polygon> region_list, Polygon region, list<Polygon> &inter_region){
//        for(auto region_item:region_list){
//            list<Polygon> output;
//            intersection(region_item, region, output);
//            appendList(inter_region,output);
//        }
//    }
//    // List和List之间的交集
//    static void listToListIntersection(list<Polygon> region1, list<Polygon> region2, list<Polygon> &inter_region){
//        for(auto region_item1:region1){
//            for(auto region_item2:region2){
//                list<Polygon> output;
//                intersection(region_item1, region_item2, output);
//                appendList(inter_region,output);
//            }
//        }
//    }

    
    // 数组转化为多边形
    static void convertPoly(Polygon poly, PolygonBoost &Poly){
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
    // 获得多边形的全部点
    static void getAllPoints(list<PolygonBoost> all_polys,Polygon &all_points){
        for(auto poly_item:all_polys){
            Polygon temp_points;
            getGemotryPoints(poly_item,temp_points);
            all_points.insert(all_points.end(),temp_points.begin(),temp_points.end());
        }
    };
    // 获得某个集合对象的全部点
    static void getGemotryPoints(PolygonBoost poly,Polygon &temp_points){
        for_each_point(poly, AllPoint<PointBoost>(&temp_points));
    };
    // 增加List链接
    template <typename T>
    static void appendList(list<T> &old_list,list<T> &new_list){
        for(auto item:new_list){
            old_list.push_back(item);
        }
    };
};
