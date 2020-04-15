//
//  main.pch
//  Nesting Problem
//
//  Created by 爱学习的兔子 on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//

#ifndef main_hpp
#define main_hpp

struct polysArrange{
    vector<vector<double>> polys; // 所有形状的情况
    vector<vector<double>> polys_position; // 所有形状的顶点位置
    vector<int> polys_orientation; // 所有形状的方向
    double width;
    double total_area;
};

#endif /* main_hpp */
