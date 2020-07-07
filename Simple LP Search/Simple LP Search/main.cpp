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
#include <cstdlib>

using namespace std;
typedef vector<vector<double>> Polygon;

class LPSearch{
protected:
    double ration_dec = 0.04,ration_inc = 0.01; // 扩大和伸缩的值
    double max_time = 30; // 允许检索的时间
    
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
    
    vector<vector<double>> all_bounds; // 全部的边界情况
    vector<Polygon> all_nfps; // 全部的nfp，通过row访问
    vector<vector<double>> all_convex_status; // 全部的凹集状态
    vector<vector<vector<double>>> all_vertical_direction; // 全部凹集处的垂线

    vector<vector<double>> pair_pd_record; // 记录重叠情况
    vector<vector<double>> miu; // 重叠调整函数
    
    string root_path = "/Users/sean/Documents/Projects/Packing-Algorithm/data/"; // 数据根目录
    
    int target_line = 1; // 目标的lp_initial的函数
public:
    LPSearch(){
        initialProblem();
    };
    void main(){
        CSVAssistant::recordSuccess(set_name, cur_length, 0.90, orientation, polys);
        return;
        shrinkBorder();
        clock_t start_time = clock();
        int search_status = 0;
        while(double(clock() - start_time)/CLOCKS_PER_SEC < max_time){
            bool feasible = minimizeOverlap();
            if(feasible == true){
                search_status = 0;
                cout << endl << "当前利用率为:" << total_area/(cur_length*width) << endl;
                best_orientation = orientation;
                best_polys = polys;
                best_length = cur_length;
                PltFunc::showPolys(polys, width, cur_length);
                shrinkBorder();
            }else{
                if(search_status == 1){
                    shrinkBorder();
                    search_status = 0;
                }else{
                    extendBorder();
                    search_status = 1;
                }
            }
            if(total_area/(best_length/width) > 0.99){
                break;
            }
        }
    };
    bool minimizeOverlap(){
        TOOLS::initial3DVector(poly_num,1,miu);
        int N = 50, it = 0;
        double Fitness = 9999999999;
       
        while(it < N){
            cout << "第" << it << "轮" <<endl;
            vector<int> permutation; randomPermutation(permutation);
//            vector<int> permutation = {11};
            for(auto choose_index: permutation){
                vector<double> top_pt;
                PackingAssistant::getTopPt(polys[choose_index], top_pt);
                double cur_pd = getIndexPD(choose_index);
                cout << choose_index << ":" << cur_pd << endl;
                if(cur_pd < bias){
                    continue;
                }
                vector<double> final_pt = top_pt;
                double final_pd = cur_pd, final_ori = orientation[choose_index];
                for(auto ori: allowed_rotation){
                    vector<double> best_pt;
                    double min_pd = lpSearch(choose_index, ori, best_pt);
                    if(min_pd < final_pd){
                        final_pd = min_pd; final_pt = best_pt; final_ori = ori;
                    }
                }
                if(final_pd < cur_pd){
                    cout << choose_index << "寻找到更优位置:" << cur_pd << "->" << final_pd;
                    polys[choose_index] = all_polygons[choose_index][final_ori];
                    PackingAssistant::slideToPoint(polys[choose_index], final_pt);
                    orientation[choose_index] = final_ori;
                    updatePD(choose_index);
                }else{
                    cout << choose_index << "没有找到更优位置" <<endl;
                }
            }
            double total_pd,max_pair_pd; getPDStatus(total_pd,max_pair_pd);
            if(total_pd < max_overlap){
                cout << endl << "结果可行" << endl;
                return false;
            }else if (total_pd < Fitness){
                Fitness = total_pd;
                it = 0;
                cout << endl << "寻找到更小重叠:" << total_pd << endl;
            }
            updateMiu(max_pair_pd);
            it ++;
            cout << endl << "当前全部重叠:" << total_pd << endl;
        }
        return true;
    };
    // 检索更优位置部分
    double lpSearch(int i,int oi, vector<double> &best_pt){
        Polygon poly = all_polygons[i][oi], ifr;
        PackingAssistant::getIFR(poly, width, cur_length, ifr);
        // 首先获得全部的NFP
        vector<Polygon> nfps;
        vector<vector<double>> nfps_convex_status, nfps_bounds;
        Polygon nfp;
        vector<double> convex_status, bounds;
        for(int j = 0; j < poly_num; j++){
            if(j == i){
                nfps_convex_status.push_back({});
                nfps_bounds.push_back({});
                nfps.push_back({});
            }
            getNFPStatus(i, j, oi, orientation[j], nfp, bounds, convex_status);
            nfps_convex_status.push_back(convex_status);
            nfps_bounds.push_back(bounds);
            nfps.push_back(nfp);
        }

        // 然后IFR切除所有的NFP计算是否仍有剩余
        vector<vector<double>> feasible_points;
        GeometryAssistant::cutIFR(ifr, nfps, feasible_points);
        if((int)feasible_points.size() > 0){
            srand((unsigned)time(NULL));
            int random_index = rand()%(int)feasible_points.size();
            best_pt = feasible_points[random_index];
            return 0;
        }
        // 计算获得所有的邻接点(IFR/NFP1/NFP2的交点)
        vector<vector<double>> all_possible_points;
        getAllPossiblePt(nfps, nfps_bounds, i, ifr, all_possible_points);
        deleteRedundancy(all_possible_points);
        
        double min_pd = 999999;
        for(auto pt: all_possible_points){
            double total_pd = 0;
            for(int j = 0; j < poly_num; j ++){
                if(j == i) continue;
                vector<double> bounds = nfps_bounds[j];
                vector<double> convex_status = nfps_bounds[j];
                Polygon nfp = nfps[j];
                if(pt[0] <= bounds[0] || pt[0] >= bounds[2] || pt[1] <= bounds[1] || pt[1] >= bounds[3] || GeometryAssistant::containPoint(nfp, pt)==false) continue;
                total_pd += getPtNFPPD(nfp, convex_status, pt) * miu[i][j];
            }
            if(total_pd < min_pd){
                min_pd = total_pd;
                best_pt = pt;
            }
        }
        return min_pd;
    };
    // 获得全部的交点和他们对应的区域
    void getAllPossiblePt(vector<Polygon> nfps, vector<vector<double>> nfps_bounds, int index, Polygon ifr, vector<vector<double>> &all_possible_points){
        for(int i = 0; i < poly_num - 1; i++){
            for(int j = i+1; j < poly_num; j++){
                if(i == index or j == index) continue;
                // xi_max < xj_min or xi_min > xj_max or yi_max < yj_min or yi_min > yj_max
                if(nfps_bounds[i][2] <= nfps_bounds[j][0] || nfps_bounds[i][0] >= nfps_bounds[j][2] || nfps_bounds[i][3] <= nfps_bounds[j][1] || nfps_bounds[i][1] >= nfps_bounds[j][3]) continue;
                GeometryAssistant::getNFPInter(nfps[i], nfps[j], ifr, all_possible_points);
            }
        }
        for(int i = 0; i < poly_num; i++){
            if(i == index) continue;
            GeometryAssistant::getNFPIFRInter(nfps[i], ifr, all_possible_points);
        }
    }
    // 删除多余的点（包括近似计算）
    void deleteRedundancy(vector<vector<double>> &target_points){
        double pt_num = (int)target_points.size();
        // 全部归化到标准差之内
        for(int i = 0; i < pt_num; i++){
            TOOLS::round(target_points[i][0],5); TOOLS::round(target_points[i][1],5);
        }
        // 删除冗余的点
        sort(target_points.begin(), target_points.end());
        vector<vector<double>> new_target_points;
        for(int i = 0; i < pt_num; i++){
            if(i == 0){
                new_target_points.push_back(target_points[i]);
                continue;
            }
            if(abs(target_points[i][0] - target_points[i-1][0]) < 0.0001 && abs(target_points[i][1] - target_points[i-1][1]) < 0.0001) continue;
            new_target_points.push_back(target_points[i]);
        }
        target_points = new_target_points;
    };
    // 更新某个形状对应的PD（平移后更新）
    void updatePD(int choose_index){
        for(int i = 0; i < poly_num; i++){
            if(i == choose_index){
                continue;
            }
            double pd = getPolysPD(choose_index, i);
            pair_pd_record[i][choose_index] = pd;
            pair_pd_record[choose_index][i] = pd;
        }
    };
    // 获得当前的全部PD和最大的PD
    void getPDStatus(double &total_pd, double &max_pair_pd){
        total_pd = 0; max_pair_pd = 0;
        for(int i = 0; i < poly_num - 1; i++){
            for(int j = i + 1; j < poly_num; j++){
                total_pd = total_pd + pair_pd_record[i][j];
                if(pair_pd_record[i][j] > max_pair_pd){
                    max_pair_pd = pair_pd_record[i][j];
                }
            }
        }
    }
    // 获得当前两两形状的PD
    double getPolysPD(int i, int j){
        // 获得基础的模型
        Polygon nfp;
        vector<double> top_pt, bounds, convex_status;
        PackingAssistant::getTopPt(polys[i], top_pt);
        getNFPStatus(i, j, orientation[i], orientation[j], nfp, bounds, convex_status);
        // 首先判断是否包含多边形（已经默认在外包多边形内部了）
        if(GeometryAssistant::containPoint(nfp, top_pt)==false) return 0;
        double pd = getPtNFPPD(nfp, convex_status, top_pt);
        return pd;
    };
    // 获得点和NFP的PD
    double getPtNFPPD(Polygon nfp, vector<double> convex_status, vector<double> pt){
        // 然后开始遍历所有的边
        double min_pd = 999999999999, pd = 0;
        vector<Polygon> all_edges;
        PackingAssistant::getPolyEdges(nfp,all_edges);
        for(auto edge:all_edges){
            vector<double> foot_pt;
            PackingAssistant::getFootPoint(pt, edge, foot_pt); // 获得底部点
            if(foot_pt[0] < min(edge[0][0],edge[1][0]) || foot_pt[0] > max(edge[0][0],edge[1][0]) || foot_pt[1] < min(edge[0][1],edge[1][1]) || foot_pt[1] > max(edge[0][1],edge[1][1])){
                continue;
            }
            pd = sqrt(pow(foot_pt[0] - pt[0],2) + pow(foot_pt[1] - pt[1],2));
            if(pd < min_pd){
                min_pd = pd;
                if(min_pd < bias){
                    return 0;
                }
            }
        }
        // 最后遍历所有的凹点
        for(int k = 0; k < (int)nfp.size(); k++){
            if(convex_status[k] == 0){
                pd = abs(pt[0]-nfp[k][0]) + abs(pt[1]-nfp[k][1]); // 计算PD
                if(pd < min_pd){
                    min_pd = pd;
                    if(min_pd < bias){
                        return 0;
                    }
                }
            }
        }
        return min_pd;
    }
    // 获得某个形状的全部PD（直接求和）
    double getIndexPD(int i){
        double total_pd = 0;
        for(int j = 0; j < poly_num; j++){
            total_pd = total_pd + pair_pd_record[i][j]*miu[i][j];
        }
        return total_pd;
    }
    // 初始化全部的PD
    void intialPairPD(){
        TOOLS::initial3DVector(poly_num, 0, pair_pd_record); // 初始化之后
        for(int i = 0; i < poly_num-1; i++){
            for(int j = i+1; j < poly_num; j++){
                double pd = getPolysPD(i, j);
                pair_pd_record[i][j] = pd;
                pair_pd_record[j][i] = pd;
            }

        }
    };
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
    // 获得对应的边界情况
    void getNFPStatus(int i, int j, int oi, int oj, Polygon &nfp, vector<double> &bounds, vector<double> &convex_status){
        // 计算行数
        int row_num = j*rotation_num*rotation_num*types_num + i*rotation_num*rotation_num + oj*rotation_num + oi;
        // 获得NFP并按照Stationary Poly的底部平移
        nfp = all_nfps[row_num];
        vector<double> bottom_pt;
        PackingAssistant::getBottomPt(polys[j], bottom_pt);
        PackingAssistant::slidePoly(nfp, bottom_pt[0], bottom_pt[1]);
        // 获得边界并按照第一个点平移
        bounds = all_bounds[row_num];
        vector<double> first_pt = nfp[0];
        bounds = {bounds[0]+first_pt[0],bounds[1]+first_pt[1],bounds[2]+first_pt[0],bounds[3]+first_pt[1]};
        // 获得凹凸状态
        convex_status = all_convex_status[row_num];
    };
    // 获得凹集状态
    void getConvexStatus(int i,int j, int oi, int oj, vector<double> &convex_status){
        int row_num = j*rotation_num*rotation_num*types_num + i*rotation_num*rotation_num + oj*rotation_num + oi;
        convex_status = all_convex_status[row_num];
    }
    // 获得随机序列
    void randomPermutation(vector<int> &permutation){
        srand((unsigned)time(NULL));
        for(int i = 0; i < poly_num; i ++){
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
        rotation_num = (int)allowed_rotation.size(); // 可以旋转的角度
        poly_num = (int)orientation.size(); // 多边形的数目
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
                vector<double> convex_status,bounds;
//                vector<vector<double>> vertical_direction;
                TOOLS::load1DVector(row["convex_status"],convex_status);
                TOOLS::load1DVector(row["bounds"],bounds);
                TOOLS::load2DVector(row["nfp"],nfp);
//                TOOLS::load2DVector(row["vertical_direction"],vertical_direction); // 由于部分情况是空集，需要进行预处理
                all_nfps.push_back(nfp);
                all_bounds.push_back(bounds);
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
                for(auto ori:allowed_rotation){
                    Polygon new_poly;
                    vector<double> new_bounds;
                    string poly_line = "o_" + to_string(ori);
                    TOOLS::load2DVector(row[poly_line],new_poly);
                    all_polygons[row_index].push_back(new_poly);
                }
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
