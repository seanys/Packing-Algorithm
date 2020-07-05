//
//  main.cpp
//  Simple LP Search
//
//  Created by 爱学习的兔子 on 2020/7/5.
//  Copyright © 2020 Yang Shan. All rights reserved.
//

#include <csv/reader.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <iostream>
#include "tools.cpp"
#include <time.h>

using namespace std;
typedef vector<vector<double>> Polygon;

class LPSearch{
protected:
    double ration_dec = 0.04,ration_inc = 0.01; // 扩大和伸缩的值
    double max_time = 300000; // 允许检索的时间
    
    string set_name; // 数据集的名称
    int poly_num; // 形状的数目
    int types_num; // 类别的中暑
    double width; // 容器的宽度
    double total_area; // 总面积计算
    double bias, max_overlap; // 差集和最大重叠
    vector<int> allowed_rotation; // 允许旋转较低
    int rotation_num; // 允许旋转较低

    double cur_length, best_length; // 当前的宽度和最佳宽度
    vector<double> orientation, best_orientation;
    vector<Polygon> polys, best_polys; // 当前的全部形状
    vector<vector<int>> polys_type; // 当前形状的类别
    
    vector<vector<Polygon>> all_polygons; // 全部的形状
    vector<vector<vector<double>>> all_bounds; // 全部的边界情况
    
    vector<Polygon> all_nfps; // 全部的nfp，通过row访问
    vector<vector<double>> all_convex_status; // 全部的凹集状态
    vector<vector<vector<double>>> all_vertical_direction; // 全部凹集处的垂线

    vector<vector<double>> pair_pd_record; // 记录重叠情况
    vector<vector<double>> miu; // 重叠调整函数
    
    string root_path = "/Users/sean/Documents/Projects/Packing-Algorithm/data/"; // 数据根目录
    
    int target_line = 70; // 目标的lp_initial的函数
public:
    LPSearch(){
        initialProblem();
    };
    void main(){
        shrinkBorder();
        minimizeOverlap();
    };
    void minimizeOverlap(){
        lpSearch(11,0);
    };
    // 检索更优位置部分
    void lpSearch(int i,int io){
        
    };
    // 获得全部的交点和他们对应的区域
    void getPtNeighbors(){
        
    }
    // 更新某个形状对应的PD（平移后更新）
    void updatePD(int choose_index){
        
    };
    // 获得当前的全部PD和最大的PD
    void getPDStatus(double &total_pd, double &max_pair_pd){
        
    }
    // 初始化全部的PD
    void intialPairPD(){
        
    };
    // 获得当前两两形状的PD
    double getPolysPD(int i, int j){
        double pd = 0;
        
        
        return pd;
    };
    // 获得某个形状的全部PD（直接求和）
    double getIndexPD(int i, vector<double> top_pt){
        double total_pd = 0;
        for(int j = 0; j < poly_num; j++){
            total_pd = total_pd + pair_pd_record[i][j]*miu[i][j];
        }
        return total_pd;
    }
    // 更新Miu的值
    void updateMiu(double max_pair_pd){
        for(int i = 0; i < poly_num; i++){
            for(int j = 0; j < poly_num; j++){
                miu[i][j] = miu[i][j] + pair_pd_record[i][j]/max_pair_pd;
                miu[j][i] = miu[j][i] + pair_pd_record[j][i]/max_pair_pd;
            }
        }
    };
    // 计算收缩后的情况
    void shrinkBorder(){
        cur_length = best_length * (1 - ration_dec);
        if(total_area/(cur_length*width) > 1){
            cur_length = total_area/width;
        }
        for(int i = 0; i < poly_num; i++){
            vector<double> right_pt;
            PackingAssistant::getRightPt(polys[i], right_pt); // 获得最右侧
            if(right_pt[0] > cur_length){
                double delta_x = cur_length - right_pt[0];
                PackingAssistant::slidePoly(polys[i], delta_x, 0);
            }
        }
        cout << endl << "当前目标利用率" << to_string(total_area/(cur_length*width)) << endl;
    };
    // 扩大边界
    void extendBorder(){
        cur_length = best_length * (1 + ration_inc);
    };
    // 根据对应的情况获得NFP
    void getNFP(int i,int j, int oi, int oj, Polygon poly_j, Polygon &nfp){
        int row_num = j*rotation_num*rotation_num*types_num + i*rotation_num*rotation_num + oj*rotation_num + oi;
        nfp = all_nfps[row_num];
        vector<double> bottom_pt;
        PackingAssistant::getBottomPt(poly_j,bottom_pt);
        PackingAssistant::slidePoly(nfp,bottom_pt[0],bottom_pt[1]);
    };
    // 获得凹集状态
    void getConvexStatus(int i,int j, int oi, int oj, vector<double> &convex_status){
        int row_num = j*rotation_num*rotation_num*types_num + i*rotation_num*rotation_num + oj*rotation_num + oi;
        convex_status = all_convex_status[row_num];
    }
    // 获得随机序列
    void randomPermutation(vector<int> &permutation){
        srand((unsigned)time(NULL));
        for(int i=0;i<poly_num;i++){
            permutation.push_back(i);
        }
        random_shuffle(permutation.begin(),permutation.end());
    };
    // 初始化整个问题
    void initialProblem(){
        csv::Reader foo;
        foo.read("/Users/sean/Documents/Projects/Packing-Algorithm/record/lp_initial.csv");
        auto rows = foo.rows(); // 读取确定的行数
        set_name = TOOLS::removeSem(rows[target_line]["set_name"]); // 数据集的名称
        width = stod(rows[target_line]["width"]); // 宽度
        bias = stod(rows[target_line]["bias"]); // PD偏差
        max_overlap = stod(rows[target_line]["max_overlap"]); // 最大的重叠
        total_area = stod(rows[target_line]["total_area"]); // 全部的面积
        types_num = stoi(rows[target_line]["types_num"]); // 形状种类
        TOOLS::load1DVector(rows[target_line]["allowed_rotation"],allowed_rotation); // 允许旋转角度
        TOOLS::load1DVector(rows[target_line]["orientation"],orientation); // 当前的形状的全部方向
        TOOLS::load1DVector(rows[target_line]["polys_type"],polys_type); // 当前的形状的全部方向
        TOOLS::load3DVector(rows[target_line]["polys"],polys); // 加载全部的形状
        best_polys = polys; // 最佳形状的复制
        best_orientation = orientation; // 最佳方向的复制
        loadAllNFP(); // 加载全部的NFP
        loadAllPolygon(); // 加载全部的形状
        cur_length = PackingAssistant::getPolysRight(polys); // 当前的形状
        best_length = cur_length; // 最佳的形状
        rotation_num = allowed_rotation.size(); // 可以旋转的角度
        poly_num = orientation.size(); // 多边形的数目
        cout << "此次检索" << set_name << "共有" << poly_num << "个形状，" << "总面积" << total_area << "，当前利用率" << total_area/(cur_length*width) << endl;
    };
    // 加载全部的形状和NFP
    void loadAllNFP(){
        csv::Reader foo;
        string path = root_path + set_name + "_nfp.csv";
        foo.read(path);
        while(foo.busy()) {
            if (foo.ready()) {
                auto row = foo.next_row();
                Polygon nfp;
                vector<double> convex_status;
                vector<vector<double>> vertical_direction;
                TOOLS::load1DVector(row["convex_status"],convex_status);
                TOOLS::load2DVector(row["nfp"],nfp);
//                TOOLS::load2DVector(row["vertical_direction"],vertical_direction); // 由于部分情况是空集，需要进行预处理
                all_nfps.push_back(nfp);
                all_convex_status.push_back(convex_status);
//                all_vertical_direction.push_back(vertical_direction);
            }
        }
    };
    // 加载全部的形状
    void loadAllPolygon(){
        csv::Reader foo;
        foo.read(root_path + set_name + "_orientation.csv");
        int row_index=0;
        while(foo.busy()) {
            if (foo.ready()) {
                auto row = foo.next_row();
                all_polygons.push_back({});
                all_bounds.push_back({});
                for(auto ori:allowed_rotation){
                    Polygon new_poly;
                    vector<double> new_bounds;
                    string poly_line = "o_" + to_string(ori);
                    string bounds_line = "b_" + to_string(ori);
                    TOOLS::load2DVector(row[poly_line],new_poly);
                    TOOLS::load1DVector(row[bounds_line],new_bounds);
                    all_polygons[row_index].push_back(new_poly);
                }
                vector <string> all_bounds = {"b_0","b_1","b_2","b_3"};
                row_index++;
            }
        }
    }
};

int main(int argc, const char * argv[]) {
    LPSearch *lp;
    lp = new LPSearch();
    lp -> main();
    
    return 0;
}
