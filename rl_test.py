'''
用于进行benchmark
'''
import numpy as np
from sequence import BottomLeftFill
max_point_num=5

def BLFwithSequence(test_path,seq_path=None):
    if seq_path!=None:
        f=open(seq_path,'r')
        seqs=f.readlines()
    data=np.load(test_path)
    size=data.shape[0]
    height=[]
    for i in range(size):
        polys_new=[]
        polys_final=[]
        line=data[i]
        if seq_path!=None:
            seq=seqs[i].split(' ')
        else:
            seq=[0,1,2,3,4,5,6,7]
            np.random.shuffle(seq)
        line=line.T
        for polys in line:
            poly=polys.reshape(max_point_num,2).tolist()
            polys_new.append(poly)
        for j in range(len(polys_new)):
            if seq_path!=None:
                index=int(seq[j][0:1])
            else:
                index=seq[j]
            polys_final.append(polys_new[index])
        blf=BottomLeftFill(1000,polys_final,vertical=True)
        height.append(blf.getLength())
    return height


if __name__ == "__main__":
    random=BLFwithSequence(r'D:\\Tongji\\Nesting\\Data\\test200_8_5.npy')
    random=np.array(random)
    np.savetxt('randomSEQ.CSV',random)

    predict=BLFwithSequence(r'D:\\Tongji\\Nesting\\Data\\test200_8_5.npy','outputs/seq2000/032823/sequence-3.csv')
    predict=np.array(predict)
    np.savetxt('predictSEQ.CSV',predict)
