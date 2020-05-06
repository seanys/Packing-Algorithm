import argparse
import os
import pprint as pp
import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import multiprocessing
import datetime
from tqdm import tqdm
from multiprocessing import Pool
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from tools.rl import NeuralCombOptRL
from tools.packing import NFPAssistant,PolyListProcessor
from heuristic import BottomLeftFill
from sequence import GA

train_preload=None
val_preload=None
training_dataset=None
val_dataset=None

class Preload(object):
    def __init__(self,source):
        self.data=np.load(source,allow_pickle=True)

    def getPolysbySeq(self,index,sequence):
        # sequence:(a,b,c...) 意思是把第a个放到第1个
        polys=self.data[index]
        polys_new=[]
        for seq in sequence:
            polys_new.append(polys[seq])
        polys_new=np.array(polys_new)
        return polys_new.tolist()

class PolygonsDataset(Dataset):
    def __init__(self,size,max_point_num,path=None,full_size=0,norm=True):
        '''
        size: 选取的数据集容量
        max_point_num: 最大点的个数
        path: 从文件加载
        full_size：数据集完整容量 若为0则对全数据集进行shuffle
        norm: 是否进行归一化
        '''
        x=[]
        self.choice=False if full_size==0 else True
        if not path:
            print('Warning: No load path')
        else:
            data=np.load(path,allow_pickle=True)
            if not self.choice:
                choice=range(size)
            else:
                choice=np.random.choice(np.array(range(full_size)),size,replace=False)
                print('Choose {} samples in total {}'.format(size,full_size))
                print('Samples like: ',choice)
            for i in choice:
                polys=data[i]
                if norm: 
                    _max=np.max(polys)
                    _min=np.min(polys)
                    med=(_max+_min)/2
                    ran=(_max-_min)
                    polys=2*(polys-med)/ran
                polys=polys.T
                x.append(polys)
            self.x=np.array(x)
            self.input=torch.from_numpy(self.x)
        # 定义数据获取顺序
        self.shuffle=np.array(range(self.__len__()))
        np.random.shuffle(self.shuffle)

    def __getitem__(self, index):
        real_index=index if self.choice else self.getRealIndex(index)
        inputs=self.input[real_index]
        return inputs

    def __len__(self):
        return len(self.input)

    def getRealIndex(self,index):
        return self.shuffle[index]

    def updateRealIndex(self):
        np.random.shuffle(self.shuffle)
        # print('Shuffled:', self.shuffle)

"""class BottomLeftFillThread (threading.Thread):
    # 多线程和多进程一起用会报错 已弃用
    def __init__(self, threadID, width, poly):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.width = width 
        self.poly = poly

    def getHeight(self,polys,width):
        '''
        polys: sample_solution
        width: 容器宽度
        利用BottomLeftFill计算Height
        '''
        try:
        #df = pd.read_csv('record/rec100_nfp.csv') # 把nfp history读入内存
        #nfp_asst=NFPAssistant(polys,load_history=True,history=df)
            bfl=BottomLeftFill(width,polys)
            return bfl.getLength()
        except:
            return 9999

    def run(self):
        # print ("开启BLF线程：" + str(self.threadID))
        self.height=self.getHeight(self.poly,self.width)
        # print ("退出BLF线程：" + str(self.threadID))

    def getResult(self):
        try:
            return self.height
        except Exception:
            return None
            
def reward_old(sample_solution, USE_CUDA=False):
    # start=datetime.datetime.now()
    # print(start,"开始reward")
    # sample_solution shape: [sourceL, batch_size, input_dim]
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    points=[]
    for i in range(n):
        sample=sample_solution[i]
        points.append(sample.cpu().numpy())
    points=np.array(points)
    result=np.zeros(batch_size)
    # threads=[] # 多线程计算BLF
    if trainning:
        p=Pool() # 多进程计算BLF
        res=[]
        for index in range(batch_size):
            poly=points[:,index,:]
            poly_new=[]
            for i in range(len(poly)):
                poly_new.append(poly[i].reshape(args['max_point_num'],2).tolist())
            poly_new=drop0(poly_new)
            nfp_asst=NFPAssistant(poly_new,load_history=True,history_path='record/{}/{}.csv'.format(args['run_name'],index+cur_batch*batch_size))
            # blf=BottomLeftFill(args['width'],poly_new,NFPAssistant=nfp_asst)
            res.append(p.apply_async(getBLF,args=(args['width'],poly_new,nfp_asst)))
            # result[index]=blf.getLength()
            # thread = BottomLeftFillThread(index,args['width'],poly_new) 
            # threads.append(thread)
        p.close()
        p.join()
        # end=datetime.datetime.now()
        # print(end,"结束reward")
        # print(end-start)
        for index in range(batch_size):
            result[index]=res[index].get()
    else: # 验证时不开多进程
        poly=points[:,0,:]
        poly_new=[]
        for i in range(len(poly)):
            poly_new.append(poly[i].reshape(args['max_point_num'],2).tolist())
        poly_new=drop0(poly_new)
        nfp_asst=NFPAssistant(poly_new,load_history=True,history_path='record/{}_val/{}.csv'.format(args['val_name'],cur_batch))
        result[0]=getBLF(args['width'],poly_new,nfp_asst)
    # for t in threads:
    #     t.start()
    # for t in threads:
    #     t.join()
    # for index in range(batch_size):
    #     result[index]=threads[index].getResult()
    return torch.Tensor(result)"""

def str2bool(v):
      return v.lower() in ('true', '1')

def getBLF(width,poly,nfp_asst):
    blf=BottomLeftFill(width,poly,NFPAssistant=nfp_asst)
    #blf.showAll()
    return blf.getLength()
        
def getGA(width,poly,nfp_asst,generations=10):
    polys_GA=PolyListProcessor.getPolyObjectList(poly,[0])
    ga=GA(width,polys_GA,nfp_asst=nfp_asst,generations=generations,pop_size=10)
    origin=ga.length_record[0]
    best=ga.global_lowest_length
    return origin-best

'''奖励函数'''
def reward(sample_solution, USE_CUDA=False):
    # start=datetime.datetime.now()
    # print(start,"开始reward")
    # sample_solution shape: [sourceL, batch_size]
    batch_size = sample_solution[0].size(0)
    result=np.zeros(batch_size)
    sequences=[]
    for sample in sample_solution:
        sequences.append(sample.numpy())
    sequences=np.array(sequences)
    if trainning:
        if batch_size>20:
            p=Pool() # 多进程计算BLF
            res=[]
            for index in range(batch_size):
                sample_id=index+cur_batch*batch_size
                real_id=training_dataset.getRealIndex(sample_id)
                sequence=sequences[:,index]
                poly_new=train_preload.getPolysbySeq(real_id,sequence)
                nfp_asst=NFPAssistant(poly_new,load_history=True,history_path='record/{}/{}.csv'.format(args['run_name'],real_id))
                res.append(p.apply_async(getBLF,args=(args['width'],poly_new,nfp_asst)))
            p.close()
            p.join()
            for index in range(batch_size):
                result[index]=res[index].get()
        else:
            for index in range(batch_size):
                sample_id=index+cur_batch*batch_size
                real_id=training_dataset.getRealIndex(sample_id)
                sequence=sequences[:,index]
                poly_new=train_preload.getPolysbySeq(real_id,sequence)
                nfp_asst=NFPAssistant(poly_new,load_history=True,history_path='record/{}/{}.csv'.format(args['run_name'],real_id))
                result[index]=getBLF(args['width'],poly_new,nfp_asst)
    else: # 验证时不开多进程
        sequence=sequences[:,0]
        real_id=val_dataset.getRealIndex(cur_batch)
        poly_new=val_preload.getPolysbySeq(real_id,sequence)
        nfp_asst=NFPAssistant(poly_new,load_history=True,history_path='record/{}_val/{}.csv'.format(args['val_name'],real_id))
        #nfp_asst=None
        result[0]=getBLF(args['width'],poly_new,nfp_asst)
    # end=datetime.datetime.now()
    # print(end,"结束reward")
    # print(end-start)
    return torch.Tensor(result)

def plot_attention(in_seq, out_seq, attentions):
    """ From http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"""

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([' '] + [str(x) for x in in_seq], rotation=90)
    ax.set_yticklabels([' '] + [str(x) for x in out_seq])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

if __name__ == "__main__":  
    multiprocessing.set_start_method('spawn',True) 
    parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

    '''数据加载'''
    parser.add_argument('--task', default='0430', help='')
    parser.add_argument('--run_name', type=str, default='reg2379')
    parser.add_argument('--val_name', type=str, default='fu')
    parser.add_argument('--train_size', default=2379, help='')
    parser.add_argument('--val_size', default=1, help='')
    parser.add_argument('--is_train', type=str2bool, default=False, help='')

    '''多边形参数'''
    parser.add_argument('--width', default=760, help='Width of BottomLeftFill')
    parser.add_argument('--max_point_num', default=4, help='')

    '''网络设计'''  
    parser.add_argument('--embedding_dim', default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_process_blocks', default=3, help='Number of process block iters to run in the Critic network')
    parser.add_argument('--n_glimpses', default=2, help='No. of glimpses to use in the pointer network')
    parser.add_argument('--use_tanh', type=str2bool, default=True)
    parser.add_argument('--tanh_exploration', default=10, help='Hyperparam controlling exploration in the pointer net by scaling the tanh in the softmax')
    parser.add_argument('--dropout', default=0., help='')
    parser.add_argument('--terminating_symbol', default='<0>', help='')
    parser.add_argument('--beam_size', default=1, help='Beam width for beam search')

    '''训练设置'''
    parser.add_argument('--batch_size', default=8, help='')
    parser.add_argument('--actor_net_lr', default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-3, help="Set the learning rate for the critic network")
    parser.add_argument('--actor_lr_decay_step', default=15000, help='')
    parser.add_argument('--critic_lr_decay_step', default=5000, help='')
    parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--reward_scale', default=2, type=float,  help='')
    parser.add_argument('--n_epochs', default=500, help='')
    parser.add_argument('--random_seed', default=24601, help='')
    parser.add_argument('--max_grad_norm', default=2.0, help='Gradient clipping')
    parser.add_argument('--use_cuda', type=str2bool, default=False, help='') # 默认禁用CUDA
    parser.add_argument('--critic_beta', type=float, default=0.7, help='Exp mvg average decay')

    # Misc
    parser.add_argument('--log_step', default=1, help='Log info every log_step steps')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--disable_tensorboard', type=str2bool, default=False)
    parser.add_argument('--plot_attention', type=str2bool, default=False)
    parser.add_argument('--disable_progress_bar', type=str2bool, default=False)

    args = vars(parser.parse_args())

    # Pretty print the run args
    pp.pprint(args)

    # Set the random seed
    torch.manual_seed(int(args['random_seed']))

    # Optionally configure tensorboard
    # if not args['disable_tensorboard']:
    #     configure(os.path.join(args['log_dir'], args['task'], args['run_name']))

    # 改用torch集成的tensorboard
    writer = SummaryWriter(os.path.join(args['log_dir'], args['task'], args['run_name']))

    size = 12 # 解码器长度（序列长度）
    input_dim = 128
    reward_fn = reward  # 奖励函数
    training_dataset = PolygonsDataset(args['train_size'],args['max_point_num'],path='{}.npy'.format(args['run_name']))
    val_dataset = PolygonsDataset(args['val_size'],args['max_point_num'],path='{}_val.npy'.format(args['val_name']))
    train_preload = Preload('{}_xy.npy'.format(args['run_name']))
    val_preload = Preload('{}_val_xy.npy'.format(args['val_name']))

    for xxx in range(0,403):
        args['load_path']='outputs/0429/reg2379/epoch-{}.pt'.format(xxx)
        '''初始化网络/测试已有网络'''
        if args['load_path'] == '':
            model = NeuralCombOptRL(
                input_dim,
                int(args['embedding_dim']),
                int(args['hidden_dim']),
                size, # decoder len
                args['terminating_symbol'],
                int(args['n_glimpses']),
                int(args['n_process_blocks']), 
                float(args['tanh_exploration']),
                args['use_tanh'],
                int(args['beam_size']),
                reward_fn,
                args['is_train'],
                args['use_cuda'])
        else:
            print('  [*] Loading model from {}'.format(args['load_path']))
            model = torch.load(
                os.path.join(
                    os.getcwd(),
                    args['load_path']
                ))
            model.actor_net.decoder.max_length = size
            model.is_train = args['is_train']    

        '''结果保存'''
        save_dir = os.path.join(os.getcwd(),
                args['output_dir'],
                args['task'],
                args['run_name'])    

        try:
            os.makedirs(save_dir)
        except:
            pass

        #critic_mse = torch.nn.MSELoss()
        #critic_optim = optim.Adam(model.critic_net.parameters(), lr=float(args['critic_net_lr']))
        actor_optim = optim.Adam(model.actor_net.parameters(), lr=float(args['actor_net_lr']))

        actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
                range(int(args['actor_lr_decay_step']), int(args['actor_lr_decay_step']) * 1000,
                    int(args['actor_lr_decay_step'])), gamma=float(args['actor_lr_decay_rate']))

        training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),shuffle=False, num_workers=4)
        validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        critic_exp_mvg_avg = torch.zeros(1)
        beta = args['critic_beta']

        if args['use_cuda']:
            model = model.cuda()
            #critic_mse = critic_mse.cuda()
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

        step = 0
        val_step = 0

        if not args['is_train']: args['n_epochs'] = '1'
        
        epoch = int(args['epoch_start'])
        for i in range(epoch, epoch + int(args['n_epochs'])):
            if args['is_train']:
                # put in train mode!
                model.train()
                trainning=True
                cur_batch=0
                avg_reward = []
                # sample_batch is [batch_size x input_dim x sourceL]
                for batch_id, sample_batch in enumerate(tqdm(training_dataloader,
                        disable=args['disable_progress_bar'])):
                    bat = Variable(sample_batch)
                    if args['use_cuda']:
                        bat = bat.cuda()
                    R, probs, actions, actions_idxs = model(bat)
                    cur_batch=cur_batch+1
                    if batch_id == 0:
                        critic_exp_mvg_avg = R.mean()
                    else:
                        critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())
                    advantage = R - critic_exp_mvg_avg
                    
                    logprobs = 0
                    nll = 0
                    for prob in probs: 
                        # compute the sum of the log probs
                        # for each tour in the batch
                        logprob = torch.log(prob)
                        nll += -logprob
                        logprobs += logprob
                
                    # guard against nan
                    nll[(nll != nll).detach()] = 0.
                    # clamp any -inf's to 0 to throw away this tour
                    logprobs[(logprobs < -1000).detach()] = 0.

                    # multiply each time step by the advanrate
                    if args['use_cuda']:
                        reinforce = advantage.cuda() * logprobs
                    else:
                        reinforce = advantage * logprobs
                    actor_loss = reinforce.mean()
                    actor_optim.zero_grad() # 清空梯度
                    actor_loss.backward() # 反向传播

                    # clip gradient norms
                    torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(),
                            float(args['max_grad_norm']), norm_type=2)

                    actor_optim.step()
                    actor_scheduler.step()

                    critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                    #critic_scheduler.step()
                    #R = R.detach()
                    #critic_loss = critic_mse(v.squeeze(1), R)
                    #critic_optim.zero_grad()
                    #critic_loss.backward()
                    
                    #torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(),
                    #        float(args['max_grad_norm']), norm_type=2)
                    #critic_optim.step()
                    
                    step += 1
                    # if not args['disable_tensorboard']:
                        # log_value('critic_loss', critic_loss.item(), step)
                    writer.add_scalar('reward', R.mean().item(), step)
                    avg_reward.append(R.mean().item())
                    writer.add_scalar('actor_loss', actor_loss.item(), step)
                    writer.add_scalar('critic_exp_mvg_avg', critic_exp_mvg_avg.item(), step)
                    # writer.add_scalar('nll', nll.mean().item(), step)
                    # if step % int(args['log_step']) == 0:
                    #     # print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(
                    #     #     i, batch_id, R.mean().item()))
                    #     example_output = []
                    #     example_input = []
                    #     for idx, action in enumerate(actions):
                    #         # if task[0] == 'tsp':
                    #         example_output.append(actions_idxs[idx][0].item())
                    #         # else:
                    #         #     example_output.append(action[0].item())  # <-- ?? 
                    #         example_input.append(sample_batch[0, :, idx][0])
                    #     #print('Example train input: {}'.format(example_input))
                    #     #print('Example train output: {}'.format(example_output))
                writer.add_scalar('avg_reward', np.mean(avg_reward), i)

            # Use beam search decoding for validation
            model.actor_net.decoder.decode_type = "beam_search"
            
            print('\n~Validating~\n')
            example_input = []
            example_output = []
            predict_sequence = []
            avg_reward = []

            # put in test mode!
            model.eval()
            trainning=False
            cur_batch=0

            for batch_id, val_batch in enumerate(tqdm(validation_dataloader,
                    disable=args['disable_progress_bar'])):
                bat = Variable(val_batch)
                if args['use_cuda']: bat = bat.cuda()
                R, probs, actions, action_idxs = model(bat)
                cur_batch=cur_batch+1
                avg_reward.append(R[0].item())
                val_step += 1.
                writer.add_scalar('val_reward', R[0].item(), int(val_step))

                if val_step % int(args['log_step']) == 0:
                    example_output = []
                    example_input = []
                    for idx, action in enumerate(actions):
                        # if task[0] == 'tsp':
                        example_output.append(action_idxs[idx][0].item())
                        # else:
                        # example_output.append(action[0].numpy())
                        example_input.append(bat[0, :, idx].cpu().numpy()) # 尝试item改numpy
                    # print('Step: {}'.format(batch_id))
                    predict_sequence.append(example_output)
                    if args['plot_attention']:
                        probs = torch.cat(probs, 0)
                        plot_attention(example_input,
                                example_output, probs.data.cpu().numpy())
            
            if not args['is_train']:
                predict_sequence=np.array(predict_sequence)
                np.savetxt(os.path.join(save_dir, 'sequence-{}.csv'.format(i)),predict_sequence,fmt='%d')
                np.savetxt(os.path.join(save_dir, 'height-{}.csv'.format(i)),avg_reward,fmt='%.05f')
                
            print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
            print('Validation overall reward var: {}'.format(np.var(avg_reward)))
            #writer.add_scalar('val_avg_reward', np.mean(avg_reward), i)
            writer.add_scalar('val_avg_reward', np.mean(avg_reward), xxx)
            # with open('rewards_val.csv',"a+") as csvfile:
            #     csvfile.write(str(i)+' '+str(np.mean(avg_reward).tolist())+' '+str(np.var(avg_reward).tolist())+'\n')
            
            if args['is_train']:
                model.actor_net.decoder.decode_type = "stochastic"
                print('Saving model...')
                torch.save(model, os.path.join(save_dir, 'epoch-{}.pt'.format(i)))

                # If the task requires generating new data after each epoch, do that here!
                # training_dataset = PolygonsDataset(args['train_size'],args['max_point_num'],path='{}.npy'.format(args['run_name']),full_size=9996)
                # training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),shuffle=False, num_workers=4)
                training_dataset.updateRealIndex()
                val_dataset.updateRealIndex()

