class bestFitPosition(object):
    def __init__(self,nfp,show):
        self.stationary=nfp.stationary
        self.sliding=nfp.sliding
        self.locus_index=nfp.locus_index
        self.nfp=nfp.nfp
        self.show=show
        self.error=1
        self.fitness=0
        self.run()

    def run(self):
        # 计算最合适的位置
        self.similar_stationary=geoFunc.similarPoly(self.stationary)
        self.sim_stationaryP=Polygon(self.similar_stationary)

        # 计算最合适位置
        aim_pt=self.getMostFit()

        if aim_pt==[0,0]:
            self.error=-6
            return
        sliding_locus=self.sliding[self.locus_index]
        geoFunc.slidePoly(self.sliding,aim_pt[0]-sliding_locus[0],aim_pt[1]-sliding_locus[1])
        if self.show==True:
            self.showAll()
    
    def showAll(self):
        pltFunc.addPolygon(self.stationary)
        pltFunc.addPolygon(self.sliding)
        pltFunc.addPolygonColor(geoFunc.similarPoly(self.sliding))
        pltFunc.addPolygonColor(self.similar_stationary)
        # pltFunc.addPolygonColor(self.nfp)
        pltFunc.showPlt()
    
    def getMostFit(self):
        '''
        获得全部边界的最值
        '''
        npf_edges=geoFunc.getPloyEdges(self.nfp)
        final_max={
            "area":0,
            "pt":[0,0]
        }
        for edge in npf_edges:
            _max=self.getEdgeFit(edge)
            if _max==[]:
                continue
            if _max["area"]>final_max["area"]:
                final_max={
                    "area":_max["area"],
                    "pt":[_max["pt"][0],_max["pt"][1]]
                }
        self.fitness=final_max["area"]
        return final_max["pt"]

    def getEdgeFit(self,edge):
        '''
        获得一条直线上的最短
        '''
        locus_pt=self.sliding[self.locus_index]
        status=[]
        # 计算两个端点的情况
        for i,pt in enumerate(edge):
            geoFunc.slidePoly(self.sliding,pt[0]-locus_pt[0],pt[1]-locus_pt[1]) # 平移到目标位置
            similar_sliding= geoFunc.similarPoly(self.sliding)
            sim_slidingP=Polygon(similar_sliding)
            if sim_slidingP.is_valid==False or self.sim_stationaryP.is_valid==False: # 排除例外情况
                return []
            inter=sim_slidingP.intersection(self.sim_stationaryP)
            inter_area=geoFunc.computeInterArea(mapping(inter))
            pt_status={
                "area":inter_area,
                "pt":[pt[0],pt[1]], # 注意该处的赋值！！！不要直接赋值！！！
                "trend":self.getTrend(edge[i],edge[1-i],inter_area)
            }
            status.append(pt_status)
        
        aim_status=self.getLargerArea(status[0],status[1])

        # 根据趋势判断变化
        if status[0]["trend"]==0 and status[1]["trend"]==0:
            return aim_status
        elif status[0]["trend"]==1 and status[1]["trend"]==0:
            return aim_status
        elif status[0]["trend"]==0 and status[1]["trend"]==1:
            return aim_status
        elif status[0]["trend"]==1 and status[1]["trend"]==1:
            mid=[(edge[0][0]+edge[1][0])/2,(edge[0][1]+edge[1][1])/2]
            status0=self.getEdgeFit([edge[0],mid])
            status1=self.getEdgeFit([mid,edge[0]])
            return self.getLargerArea(status0,status1)
        else:
            print("##########数据异常#########")
    
    def getTrend(self,pt0,pt1,_area):
        '''
        计算在某个一个方向上的变化趋势
        '''
        vec=[pt1[0]-pt0[0],pt1[1]-pt0[1]]
        _len=math.sqrt(math.pow(vec[0],2)+math.pow(vec[1],2)) # 一般情况下都是大于1
        if _len<2.5:
            return 0 # 直接返回降低，两个位置都是降低
        
        # 平移一下
        mid=[pt0[0]+vec[0]/_len,pt0[1]+vec[1]/_len]
        locus_pt=self.sliding[self.locus_index]
        geoFunc.slidePoly(self.sliding,mid[0]-locus_pt[0],mid[1]-locus_pt[1])

        # 计算平移后的交叉面积
        similar_sliding= geoFunc.similarPoly(self.sliding)
        sim_slidingP=Polygon(similar_sliding)
        inter=sim_slidingP.intersection(self.sim_stationaryP)
        inter_area=geoFunc.computeInterArea(mapping(inter))

        # 如果更大，就是1，如果更小或者不变，就是0
        if inter_area>_area:
            return 1 # 增加
        else:
            return 0 # 降低，不考虑不变的情况

    def getLargerArea(self,status0,status1):
        # print("status0",status0)
        # print("status1",status1)
        if status0["area"]>status1["area"]:
            return status0
        else:
            return status1