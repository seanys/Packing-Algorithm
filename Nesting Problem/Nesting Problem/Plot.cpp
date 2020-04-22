//
//  Plot.cpp
//  Nesting Problem
//
//  Created by Yang Shan on 2020/4/14.
//  Copyright © 2020 Tongji SEM. All rights reserved.
//
//  Matplotlib Introduction
//  https://readthedocs.org/projects/matplotlib-cpp/downloads/pdf/latest/
//  Give up this parts temporarily

#include <iostream>
//#include <matplotlibcpp.h>
#include "../tools/matplotlib.h"
#include <vector>

namespace plt = matplotlibcpp;
using namespace std;

class PltFunc{
public:
    static double pltTest(){
//        PyObject * listObj;
        vector<double> x = {1, 2, 3, 4};
        vector<double> y = {1, 4, 9, 16};
        plt::plot(x, y);
        plt::show();
//        plt::savefig("minimal.pdf");
        
        return 0;
    }
};
