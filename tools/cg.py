'''
Comlumn Generation Python 版本
'''
from pulp import *
import random

class MasterProblem(object):
    '''
    修改来源：https://gist.github.com/Bart6114/8414730
    '''
    def __init__(self,maxValue,itemLengths,itemDemands,initialPatterns,problemname):
		self.maxValue=maxValue
		self.itemLengths=itemLengths
		self.itemDemands=itemDemands
		self.initialPatterns=initialPatterns
		
		self.prob = LpProblem(problemname,LpMinimize)	# set up the problem
		
		self.obj = LpConstraintVar("obj")   # generate a constraint variable that will be used as the objective
		self.prob.setObjective(self.obj)
		
		self.PatternVars=[]
		self.constraintList=[]   # list to save constraint variables in
		for i,x in enumerate(itemDemands):		# create variables & set the constraints, in other words: set the minimum amount of items to be produced
			var=LpConstraintVar("C"+str(i),LpConstraintGE,x)  # create constraintvar and set to >= demand for item
			self.constraintList.append(var)
			self.prob+=var
		
			
		for i,x in enumerate(self.initialPatterns):  #save initial patterns and set column constraints 
			temp=[]
			for j,y in enumerate(x):
				if y>0: 
					temp.append(j)
			
			
			var=LpVariable("Pat"+str(i)	, 0, None, LpContinuous, lpSum(self.obj+[self.constraintList[v] for v in temp]))  # create decision variable: will determine how often pattern x should be produced
			self.PatternVars.append(var)
		
		
	def solve(self):
		self.prob.writeLP('prob.lp')
		self.prob.solve()  # start solve
		
		return [self.prob.constraints[i].pi for i in self.prob.constraints]
		
			
	def addPattern(self,pattern):  # add new pattern to existing model
		
		self.initialPatterns.append(pattern)
		temp=[]
		
		for j,y in enumerate(pattern):
			if y>0: 
				temp.append(j)
		
		var=LpVariable("Pat"+str(len(self.initialPatterns))	, 0, None, LpContinuous, lpSum(self.obj+[pattern[v]*self.constraintList[v] for v in temp]))
		self.PatternVars.append(var)
		
	
	def startSlave(self,duals):  # create/run new slave and return new pattern (if available)
		
		newSlaveProb=SlaveProblem(duals,self.itemLengths,self.maxValue)
				
		pattern=newSlaveProb.returnPattern()
		return pattern
		
	def setRelaxed(self,relaxed):  # if no new patterns are available, solve model as IP problem
		if relaxed==False:
			for var in self.prob.variables():
				var.cat = LpInteger
			
	def getObjective(self):
		return value(self.prob.objective)
		
	def getUsedPatterns(self):
		usedPatternList=[]
		for i,x in enumerate(self.PatternVars):
			if value(x)>0:
				usedPatternList.append((value(x),self.initialPatterns[i]))
		return usedPatternList

class SlaveProblem(object):
    def __init__(self,duals, itemLengths,maxValue):
		self.slaveprob=LpProblem("Slave solver",LpMinimize)
        self.varList=[LpVariable('S'+str(i),0,None,LpInteger) for i,x in enumerate(duals)]
        self.slaveprob+=-lpSum([duals[i]*x for i,x in enumerate(self.varList)])  #use duals to set objective coefficients
        self.slaveprob+=lpSum([itemLengths[i]*x for i,x in enumerate(self.varList)])<=maxValue 

        self.slaveprob.writeLP('slaveprob.lp')
        self.slaveprob.solve() 
        self.slaveprob.roundSolution() #to avoid rounding problems

        

    def returnPattern(self):
        pattern=False
        if value(self.slaveprob.objective) < -1.00001:
            pattern=[]
            for v in self.varList:
                pattern.append(value(v))
        return pattern

def colunmGeneration():
    random.seed(2012)

    nrItems=12
    lengthSheets=20

    itemLengths=[]
    itemDemands=[]

    while len(itemLengths)!=nrItems:
	    length=random.randint(5, lengthSheets-2)
	    demand=random.randint(5, 100)
	    if length not in itemLengths:
		    itemLengths.append(length)
		    itemDemands.append(demand)
	
    print("Item lengts  : %s" % itemLengths)
    print("Item demands : %s\n\n" % itemDemands)

    patterns=[]
    print("Generating start patterns:")
    ## generate simple start patterns
    for x in range(nrItems):
	    temp=[0.0 for y in range(x)]
	    temp.append(1.0)
	    temp+=[0.0 for y in range(nrItems-x-1)]
	    patterns.append(temp)
	    print temp

    print("\n\nTrying to solve problem")
    CGprob=MasterProblem(lengthSheets,itemLengths,itemDemands,patterns,'1D cutting stock')
	
    relaxed=True
    while relaxed==True:   # once no more new columns can be generated set relaxed to False
	    duals=CGprob.solve()
	
    	newPattern=CGprob.startSlave(duals)
	
    	print('New pattern: %s' % newPattern)
	
    	if newPattern:
	    	CGprob.addPattern(newPattern)
    	else:
	    	CGprob.setRelaxed(False)
		    CGprob.solve()
    		relaxed=False

    print("\n\nSolution: %s sheets required" % CGprob.getObjective())

    t=CGprob.getUsedPatterns()
    for i,x in enumerate(t):
	    print("Pattern %s: selected %s times	%s" % (i,x[0],x[1]))