from pulp import *
import random

def sovleLP(a,b,c):
    # 变量所有的变量
    all_var=[] 
    for i in range(len(c)):
        all_var.append(LpVariable("var"+str(i),0))

    # 初始化限制
    prob = LpProblem("Minimize",LpMinimize)   

    # 定义目标函数
    prob += lpSum([c[i]*all_var[i] for i in range(len(c))])

    # 定义约束函数
    for j in range(len(a)):
        prob += lpSum([a[j][i]*all_var[i] for i in range(len(c))]) >= b[j]

    prob.solve()

    result=[]
    for v in prob.variables():
        result.append(v.varValue)
        print(v.name, "=", v.varValue)
    print("Total Cost of Ingredients per can = ", value(prob.objective))
    
    return result


if __name__=='__main__':
    # a=[[10,0,0],[0,10,0],[0,0,10]]
    # b=[10,10,10]
    # c=[1,4,10]
    # sovleLP(a,b,c)
	colunmGeneration()