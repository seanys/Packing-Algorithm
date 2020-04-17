## 2D Irregular Bin Packing 

### 介绍

Literature Review/Tutorial：[排样问题文献综述/教程](https://seanys.github.io/2020/03/17/排样问题综述/) 作者 Yang shan 

English Version：https://github.com/seanys/Packing-Algorithm/blob/master/readme_en.md editing

Author: Yang Shan, Wang Zilu (Department of Science and Managment, Tongji University)

### 数据库

EURO Dataset：https://www.euro-online.org/websites/esicup/data-sets/#1535972088237-bbcb74e3-b507

## 算法实现情况

### 版本

Python：https://github.com/seanys/Packing-Algorithm

C++：https://github.com/seanys/Packing-Algorithm/tree/master/Nesting%20Problem

### 基础算法

- [x] **No-fit Polygon**：基本实现，Start Point的部分暂未实现，参考论文 Burke E K , Hellier R S R , Kendall G , et al. Complete and Robust No-Fit Polygon Generation for the Irregular Stock Cutting Problem[J]. European Journal of Operational Research, 2007, 179(1):27-49.

形状A固定位置，在B上选择一个参考点比如左下角点P，形状B紧贴着绕A一周，P的轨迹会形成一个形状即NFP，P在NFP上或外部，则该排列可行；如果P点在NFP内，则方案不可行（[图片来源](https://github.com/Jack000/SVGnest)）

<img src="https://camo.githubusercontent.com/1156f6f8323c52dea2981604dd780b02add19e86/687474703a2f2f7376676e6573742e636f6d2f6769746875622f6e66702e706e67" alt="img" style="width:50%;" />



### 序列排样

- [x] Bottom Left Fill：已经实现，参考论文 

a. 选择一个形状加入，通过计算inner fit polygon，也就是形状绕着Bin/Region内部一周，参考点P会形成的一个长方形，P点在该长方形内部则解是feasible solution 

b. 选择能够摆到的最左侧位置，放进去即可 （[图片来源](https://github.com/Jack000/SVGnest)）

<img src="https://camo.githubusercontent.com/f7973d894432676e37c3489c3248c3a31cf3e945/687474703a2f2f7376676e6573742e636f6d2f6769746875622f6e6670322e706e67" alt="No Fit Polygon example" style="width:50%;" />

- [x] TOPOS：已经实现，参考论文：
- [x] GA/SA：两个优化算法优化顺序已经实现

### 基于布局的优化

- [x] Fast Neighborhood Search：基本实现，有一些Bug还需要修改
- [x] Cuckoo Search：基本算法已经实现，需要进一步完善
- [x] ~~ILSQN：不准备写了，没太大意义~~
- [x] ~~Guided Local Search：同上~~

### 线性规划排样

- [x] Compaction：压缩边界，已经实现
- [x] Separation：去除重叠，已经实现
- [ ] SHAH：基于模拟退火算法和上述两个算法的Hybrid Algorithm，暂时未做



## 研究一 线性规划检索最优解

### 研究原因



### 算法方案



### 实现情况

Python版本已经实现，有一些问题外加计算速度比较慢，正在筹备C++版本





## 研究二 通过神经网络获得最优解

### 研究意义

排样问题在这里指的是在给定一个区域(region)的宽度和底部的情况下，向其中添加给定的形状，保证形状均在区域内且没有重叠(overlap)并使得用料最少

有两种传统方案解决该问题：

**方案一**：按照比如面积递减原则，设置多个样片的添加顺序(sequence)，按照该顺序添加形状即可获得初始解，通过修改添加顺序以获得更优解

<img src="https://tva1.sinaimg.cn/large/006tNbRwly1gazftkphzyj30sq0i20w7.jpg" alt="image-20200117121804453" style="width:50%;" />

**方案二**：把形状全部加入固定的区域(region)以获得初始解/可行解，通过对部分形状的重定位(relocate) 来获得更优结果

<img src="https://tva1.sinaimg.cn/large/006tNbRwly1gazft8gnwmj30ki0mwahk.jpg" alt="image-20200117121742525" style="width:50%;" />

<img src="https://tva1.sinaimg.cn/large/006tNbRwly1gazftr0bfxj31fa0ksdsh.jpg" alt="image-20200117121814492" style="width:50%;" />



**两个方案的思路都是在获得初始解的情况下进行优化，初始解的优劣都直接决定了所需要的优化次数**。然而现在的初始解的获得方法比较单一，比如面积递减、大形状靠左下角等，如果能够通过神经网络直接对所有的形状进行处理，可以直接获得更优的序列(sequence)或布局(arrangement)。

**RNN等序列学习网络在获得更优序列是可行**的：阿里巴巴已经通过RNN的衍生网络Pointer network和以及强化学习算法Q-Learning，对排箱问题进行研究，可以获得更好效果，网络的输入是箱子的长宽高，输出是新的序列。

**神经网络可以避免陷入局部最优**：传统的算法本质上来说是贪婪算法，比如按照不同的顺序加入形状，每次放置只考虑已经加入的形状的情况，这就造成了后续的最优化可能会陷入局部最优，但是如果通过网络处理，是可以直接考虑全部的形状情况的，阿里巴巴的实验结果也验证了这一点。

**Shape Signature算法可将二维形状转为Vector以作为网络输入**：二维排样最难的是不规则形状排样，需要将其Encode为一位向量才可作为网络的输入，Shape Signature算法满足了这个要求，我们也对其进行了改进，在标准数据集上都获得比较好的效果。



**综上所述，我们完全也可以通过Neural Network来对二维排样问题进行学习**

(1) 通过RNN等序列学习的网络，在方案一(optimize sequence)的基础上

(2) 通过BP Neural Network，在方案二(relocate arrangement)的基础上，直接获得更优的初始布局



### 通过神经网络学习排样

#### **Encode算法**

a. 网络输入需要是一维数组，encode就是降维过程。

b. 三维排箱问题的箱子可以通过长宽高直接表示，即将三维降低到一维`[length,width,height]`。

c. 二维的形状如果需要作为输入，同样也需要降低到一维，这里采用了shape signature算法：从centroid向不同方向发出射线，射线与边界有交点，为该角度上的feature

#### 数据集建立

较为复杂，正在研究中。现阶段实现了较为简单形状的预测，但是暂时无法推广到更多的形状。理论上只有增加encode的损失，降低精度才可能推广到普遍情况。



