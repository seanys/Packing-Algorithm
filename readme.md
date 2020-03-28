## 2D Irregular Bin Packing 

### 介绍

Blog: https://seanys.github.io

English Version：https://github.com/seanys/Packing-Algorithm/blob/master/readme_en.md editing

经典算法完成版本：https://github.com/seanys/2D-Irregular-Packing-Algorithm

Author: Yang Shan, Wang Zilu (Department of Science and Managment, Tongji University)

### 命名规则

类名：首字母大写如CuckooSearch

函数名：首字母小写getResult

变量命名：常量全部大写如MAX_VALUE，非常量全部小写max_value



## 预测效果

### 说明

蓝色：预测结果

黑色：最优结果（似乎有点问题有些不是最优）

### 四个样片

#### 预测后可行化结果

![prediction_vise_1793](https://tva1.sinaimg.cn/large/0082zybpgy1gc26oe1uqvj30zk0qogo1.jpg)

![prediction_revise_1340](https://tva1.sinaimg.cn/large/0082zybpgy1gc657ctte8j30zk0qodht.jpg)

![prediction_revise_1731](https://tva1.sinaimg.cn/large/0082zybpgy1gc272w3gyxj30zk0qoq5c.jpg)

#### 直接预测结果

![prediction_1391](https://tva1.sinaimg.cn/large/0082zybpgy1gc26hk5aywj30zk0qotaw.jpg)



## 任务安排

- [ ] 1. [数据集-羊山] 分别采用GA建立数据集并测试网络
- [ ] 2. [数据集-王子路] 采用Shirk算法建立可行解和初始解
- [ ] 3. [网络训练-羊山] 应用2. 的算法建立benchmark进行比较
- [ ] 4. [数据集优化-羊山]采用DJD取代Bottom Left建立更好的数据集
- [ ] 5. [网络训练-王子路]通过强化学习与Pointer Network尝试学习序列

其他：NFP可能存在部分问题需要优化、注意学习论文写作、统计和ML的相关基础知识的学习



## 通过神经网络获得最优解

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

### **进一步计划安排**

**第一步：推广实现可以适用所有形状的训练集**：现阶段实现了特定类别的形状输入，可以获得组合结果，后续需要扩大到任意常见形状的输入都可以实现组合

**第二步：实现使用NN预测整体的初始布局**，即在方案二的基础上优化初始解，可以先实现规模较小的，再实现规模较大的

- 少量样片的情况(>5)

- 较多样片的情况(>20)

**第三步：通过RNN/Pointer Network进行样片的序列预测**，即在方案一的基础上优化初始解，也主要分为两阶段

- 少量样片的情况(>5)
- 较多样片的情况(>20)

### 排样算法说明

#### 方案一

摘录于：https://github.com/Jack000/SVGnest

**1.基础算法[No fit polygon](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.440.379&rep=rep1&type=pdf)**

形状A固定位置，在B上选择一个参考点比如左下角点P，形状B紧贴着绕A一周，P的轨迹会形成一个形状即NFP，P在NFP上或外部，则该排列可行；如果P点在NFP内，则方案不可行

<img src="https://camo.githubusercontent.com/1156f6f8323c52dea2981604dd780b02add19e86/687474703a2f2f7376676e6573742e636f6d2f6769746875622f6e66702e706e67" alt="img" style="width:50%;" />



**2.正式放置形状。**

表示方法：所有的多边形采用顶点的绝对坐标表示（相对于左下角的坐标）

添加方式：

a. 选择一个形状加入，通过计算inner fit polygon，也就是形状绕着Bin/Region内部一周，参考点P会形成的一个长方形，P点在该长方形内部则解是feasible solution 

b. 选择一个判断标准，比如gravity-lowest重心最低还有bottom-left-fill，在IFP中选择y坐标最低或者|x|+|y|最小的点，移过去即可 

c. 如果是添加第n个(n>2)，则需要先计算和已经添加进去的形状的NFP，如图为NFP1/NFP2，NFP取并集，P在外边界的外部可行，如第二张图；其次计算IFP，IFP内部可行。采用shapely库，即`IFP.difference(NFP)` 所获的形状内部可行

<img src="https://camo.githubusercontent.com/f7973d894432676e37c3489c3248c3a31cf3e945/687474703a2f2f7376676e6573742e636f6d2f6769746875622f6e6670322e706e67" alt="No Fit Polygon example" style="width:50%;" />



优化方法：a. 采用面积递减原则设置初始的加入顺序，按照该顺序加入形状 b. 采用遗传算法对加入的序列进行调整，从而获得更优解

#### 方案二

主要通过shrink和relocate实现，具体操作不太了解

### 通过神经网络学习排样

#### **Encode算法**

a. 网络输入需要是一维数组，encode就是降维过程。

b. 三维排箱问题的箱子可以通过长宽高直接表示，即将三维降低到一维`[length,width,height]`。

c. 二维的形状如果需要作为输入，同样也需要降低到一维，这里采用了shape signature算法：从centroid向不同方向发出射线，射线与边界有交点，为该角度上的feature

#### 数据集建立

较为复杂，正在研究中。现阶段实现了较为简单形状的预测，但是暂时无法推广到更多的形状。理论上只有增加encode的损失，降低精度才可能推广到普遍情况。

#### 网络训练（方案一）

Input：Encode(Polygons)

Network: RNN/Pointer Network

Output: Sequence of Polygons

输入形状，输出序列

#### 网络训练（方案二）

Input：Encode(Polygons)

Network: 最简单的BP神经网络

Output: Position of centroid-Decode



