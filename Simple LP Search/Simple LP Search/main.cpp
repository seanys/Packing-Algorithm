//
//  main.cpp
//  Simple LP Search
//
//  Created by 爱学习的兔子 on 2020/7/5.
//  Copyright © 2020 Yang Shan. All rights reserved.
//

#include <iostream>
#include <vector>
#include <stdbool.h>
#include <csv/reader.hpp>

class LPSearch{
protected:
    VectorPoints cur_ifr; // 获得当前的IFR
    double poly_num; // 形状总数
    vector<VectorPoints> all_nfps; // 获得当前全部NFP
    vector<Polygon> all_nfps_poly; // 获得当前全部NFP的Polygon模型
    vector<vector<vector<double>>> edges_target_funs,points_target_funs; // 分别获得直线和点的目标函数
    
    PolysArrange cur_solution; // 记录当前解，加载读取时使用
    
    vector<VectorPoints> best_polys,polys; // 暂存当前的全部形状
    vector<int> best_orientation,cur_orientation; // 当前最佳情况和当前情况采用的方向
    double best_length,cur_length; // 当前宽度和最佳宽度

    vector<int> polys_type; // 所有形状情况（不变）
    double width; // 放置宽度（不变）

    vector<vector <double>> poly_overlap; // 重叠情况
    vector<vector <double>> nfp_overlap_pair; // NFP的重叠情况
    vector<vector <double>> nfp_overlap_list; // 每个NFP的重叠对象（计算相交）
    vector<vector <double>> miu; // 重叠情况
    
    vector<vector <VectorPoints>> all_polygons; // 全部形状的加载
    
    int choosed_index; // 当前选择的index
        
    NFPAssistant *nfp_assistant; // NFP辅助操作
    
    vector<list<Polygon>> nfp_sub_ifr; // 除去IFR后的NFP，全部存为Vector<VectorPoints>（计算相交情况）
    vector<vector<list<Polygon>>> nfp_sub_ifr_phase; // 各个阶段的NFP的记录
    vector<vector<vector<int>>> target_indexs; // 获得各个阶段交集的目标区域
public:
        LPSearch(){
            DataAssistant::readAllPolygon(all_polygons); // 加载各个方向的形状
            DataAssistant::readBLF(3,cur_solution); // 读取BLF的解
            nfp_assistant = new NFPAssistant("/Users/sean/Documents/Projects/Data/fu_clock.csv",cur_solution.type_num,4); // NFP辅助
            
            polys = cur_solution.polys; // 当前的形状，记录形状变化
            cur_orientation = cur_solution.polys_orientation; // 赋值全部形状（变化）
            cur_length = PackingAssistant::arrangetLenth(polys); // 当前长度（变化）
            
            best_polys = cur_solution.polys; // 最佳结果存储
            best_length = cur_length; // 最佳长度（变化）
            best_orientation = cur_solution.polys_orientation; // 最佳长度（变化）

            width = cur_solution.width; // 宽度（不变）
            poly_num = cur_solution.total_num; // 形状总数（不变）
            polys_type = cur_solution.polys_type; // 形状类别（不变）
            
        };
        void run(){
            // 限制检索时间是否超过时间
            double ration_dec=0.04,ration_inc=0.01;
    //        double max_time=1200;
            
            shrinkBorder(ration_dec);
            
            // 执行最小化重叠后，判断解是否可行
            minimizeOverlap();
            
        };
        // 计算收缩后的情况
        void shrinkBorder(double ration_dec){
            cur_length = best_length*(1 - ration_dec);
            for(int i=0;i<poly_num;i++){
                vector<double> right_pt;
                PackingAssistant::getRightPt(polys[i],right_pt);
                if(right_pt[0]>cur_length){
                    PackingAssistant::slidePoly(polys[i], cur_length-right_pt[0], 0);
                }
            }
        };
        // 扩大边界
        void extendBorder(double ration_inc){
            cur_length = best_length*(1 - ration_inc);
        };
};

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}
