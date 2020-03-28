'''
    This file is used for result analysis
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import csv
import numpy as np

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

class staAnalysis(object):
    def __init__(self):
        print("begin")
        # self.data=pd.read_csv("/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/arbitrary_data/original_rectangle_triangle/original_rectangle_triangle.csv")
        # self.countData()
        # self.delRebound()
        # self.correlationFun()
        # self.distriFunExcess("/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/euro_process/rebuild_res.csv")
        # self.distriFunContain("/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/euro_process/rebuild_res.csv")
        self.graphStatistic("/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/euro_process/rebuild_res.csv")
    
    def distriFunExcess(self,_path):
        '''
        包含和存在率的分布情况
        '''
        df = pd.read_csv(_path)
        plt.figure(figsize=(13,10), dpi= 80)
        sns.distplot(df.loc[df['Contain Ratio'] >0, "Excess Ratio"], color="dodgerblue", label="excess ratio", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})

        # plt.title('The Ratio of the Area exceed Original Region in Reconstruction Result', fontsize=22)
        plt.legend()
        plt.show()
    
    def correlationFun(self):
        '''
        相关分布：结果显示基本没有相关性
        '''
        df = pd.read_csv("/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/train_data/rebuild_res_centroid_clean.csv")
        # df = pd.read_csv("/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/standard_data/rebuild_res.csv")
        # Import Data
        # df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")

        # Each line in its own column
        sns.set_style("white")
        gridobj = sns.lmplot(x="contain_ratio", y="exceed_ratio", 
            data=df, 
            height=7, 
            robust=True, 
            palette='Set1', 
            scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))

        plt.title('Correlation of contain_ratio and exceed_ratio', fontsize=22)
        plt.legend()
        plt.show()

    def countData(self):
        '''
        根据数据结果统计情况
        '''
        contain_ratio=[0,0,0,0,0,0]
        exceed_ratio=[0,0,0,0]
        for index, row in self.data.iterrows():
            if row['contain_ratio1']>0.99:
                contain_ratio[0]=contain_ratio[0]+1
            elif 0.98<row['contain_ratio1']<=0.99:
                contain_ratio[1]=contain_ratio[1]+1
            elif 0.97<row['contain_ratio1']<=0.98:
                contain_ratio[2]=contain_ratio[2]+1
            elif 0.96<row['contain_ratio1']<=0.97:
                contain_ratio[3]=contain_ratio[3]+1
            elif 0.95<row['contain_ratio1']<=0.96:
                contain_ratio[4]=contain_ratio[4]+1
            else:
                contain_ratio[5]=contain_ratio[5]+1

        for index, row in self.data.iterrows():
            if row['exceed_ratio1']<0.001:
                exceed_ratio[0]=exceed_ratio[0]+1
            elif 0.001<row['exceed_ratio1']<=0.01:
                exceed_ratio[1]=exceed_ratio[1]+1
            elif 0.01<row['exceed_ratio1']<=0.05:
                exceed_ratio[2]=exceed_ratio[2]+1
            else:
                exceed_ratio[3]=exceed_ratio[3]+1
                
    def delRebound(self):
        '''
        删除重复/冗余数据
        '''
        rebuild_res=[]
        for index, row in self.data.iterrows():
            contain_ratio=row['contain_ratio']
            exceed_ratio=row['exceed_ratio']
            inter_area=row['inter_area']
            add=True
            for res in rebuild_res:
                if res[0]==contain_ratio and res[1]==exceed_ratio:
                    add=False
            if add==True:
                rebuild_res.append([contain_ratio,exceed_ratio,inter_area])
        
        with open("/Users/sean/Documents/Projects/Algorithm Learn/Packing Algorithm/train_data/rebuild_res_centroid_clean.csv","a+") as csvfile:
            writer = csv.writer(csvfile)
            _len=len(rebuild_res)
            for i in range(_len):
                writer.writerows([[rebuild_res[i][0],rebuild_res[i][1],rebuild_res[i][2]]])
                
if __name__ == '__main__':
    # tr=textReader()
    # tr.polyShow()
    sa=staAnalysis()
    # de=dataEvaluate()