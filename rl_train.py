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
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from tensorboard_logger import configure, log_value
from tools.rl import NeuralCombOptRL
from heuristic import BottomLeftFill,NFPAssistant
from rl_test import generateData_fu

class PolygonsDataset(Dataset):
    def __init__(self,size,max_point_num,path=None):
        '''
        size: 数据集容量
        max_point_num: 最大点的个数
        path: 从文件加载
        '''
        x=[]
        if not path:
            for i in range(size):
                polys=generateData_fu(10)
                polys=polys.T
                x.append(polys)
            self.x=np.array(x)
            self.input=torch.from_numpy(self.x)
        else:
            data=np.load(path)
            for i in range(size):
                x.append(data[i])
            self.x=np.array(x)
            self.input=torch.from_numpy(self.x)

    def __getitem__(self, index):
        inputs=self.input[index]
        return inputs

    def __len__(self):
        return len(self.input)

    def save_data(self,path):
        np.save(path,self.x)


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
            bfl=BottomLeftFill(width,polys,vertical=True)
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
            return None"""

def str2bool(v):
      return v.lower() in ('true', '1')

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

    '''数据加载'''
    parser.add_argument('--task', default='0401', help='')
    parser.add_argument('--batch_size', default=8, help='')
    parser.add_argument('--train_size', default=1000, help='')
    parser.add_argument('--val_size', default=1000, help='')

    '''多边形参数'''
    parser.add_argument('--width', default=1000, help='Width of BottomLeftFill')
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
    parser.add_argument('--actor_net_lr', default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--actor_lr_decay_step', default=5000, help='')
    parser.add_argument('--critic_lr_decay_step', default=5000, help='')
    parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--reward_scale', default=2, type=float,  help='')
    parser.add_argument('--is_train', type=str2bool, default=True, help='')
    parser.add_argument('--n_epochs', default=500, help='')
    parser.add_argument('--random_seed', default=24601, help='')
    parser.add_argument('--max_grad_norm', default=2.0, help='Gradient clipping')
    parser.add_argument('--use_cuda', type=str2bool, default=False, help='') # 默认禁用CUDA
    parser.add_argument('--critic_beta', type=float, default=0.9, help='Exp mvg average decay')

    # Misc
    parser.add_argument('--log_step', default=1, help='Log info every log_step steps')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--run_name', type=str, default='fu1000')
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
    if not args['disable_tensorboard']:
        configure(os.path.join(args['log_dir'], args['task'], args['run_name']))

    size = 10 # 解码器长度（序列长度）

    '''奖励函数'''
    def reward(sample_solution, USE_CUDA=False):
        # sample_solution shape: [sourceL, batch_size, input_dim]
        batch_size = sample_solution[0].size(0)
        n = len(sample_solution)
        height = Variable(torch.zeros([batch_size]))
        points=[]
        for i in range(n):
            sample=sample_solution[i]
            points.append(sample.cpu().numpy())
        points=np.array(points)
        result=np.zeros(batch_size)
        # threads=[] # 多线程计算BFL
        for index in range(batch_size):
            poly=points[:,index,:]
            poly_new=[]
            for i in range(len(poly)):
                poly_new.append(poly[i].reshape(args['max_point_num'],2).tolist())
            bfl=BottomLeftFill(args['width'],poly_new,vertical=True)
            result[index]=bfl.getLength()
            # thread = BottomLeftFillThread(index,args['width'],poly_new) 
            # threads.append(thread)
        # for t in threads:
        #     t.start()
        # for t in threads:
        #     t.join()
        # for index in range(batch_size):
        #     result[index]=threads[index].getResult()
        return torch.Tensor(result)

    def plot_attention(in_seq, out_seq, attentions):
        """ From http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"""

        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions, cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([' '] + [str(x) for x in in_seq], rotation=90)
        ax.set_yticklabels([' '] + [str(x) for x in out_seq])

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()

    input_dim = 8
    reward_fn = reward  # 奖励函数
    training_dataset = PolygonsDataset(args['train_size'],args['max_point_num'])
    val_dataset = PolygonsDataset(args['val_size'],args['max_point_num'],path='fu1000_10_5.npy')
    # print(val_dataset.input)

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

    training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),shuffle=True, num_workers=4)

    validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

    critic_exp_mvg_avg = torch.zeros(1)
    beta = args['critic_beta']

    if args['use_cuda']:
        model = model.cuda()
        #critic_mse = critic_mse.cuda()
        critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

    step = 0
    val_step = 0

    if not args['is_train']:
        args['n_epochs'] = '1'
    

    epoch = int(args['epoch_start'])
    for i in range(epoch, epoch + int(args['n_epochs'])):
        
        if args['is_train']:
            # put in train mode!
            model.train()
            # sample_batch is [batch_size x input_dim x sourceL]
            for batch_id, sample_batch in enumerate(tqdm(training_dataloader,
                    disable=args['disable_progress_bar'])):
                bat = Variable(sample_batch)
                if args['use_cuda']:
                    bat = bat.cuda()
                R, probs, actions, actions_idxs = model(bat)
        
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
                
                if not args['disable_tensorboard']:
                    log_value('avg_reward', R.mean().item(), step)
                    log_value('actor_loss', actor_loss.item(), step)
                    #log_value('critic_loss', critic_loss.item(), step)
                    log_value('critic_exp_mvg_avg', critic_exp_mvg_avg.item(), step)
                    log_value('nll', nll.mean().item(), step)

                if step % int(args['log_step']) == 0:
                    # print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(
                    #     i, batch_id, R.mean().item()))
                    example_output = []
                    example_input = []
                    for idx, action in enumerate(actions):
                        # if task[0] == 'tsp':
                        example_output.append(actions_idxs[idx][0].item())
                        # else:
                        #     example_output.append(action[0].item())  # <-- ?? 
                        example_input.append(sample_batch[0, :, idx][0])
                    #print('Example train input: {}'.format(example_input))
                    #print('Example train output: {}'.format(example_output))

        # Use beam search decoding for validation
        model.actor_net.decoder.decode_type = "beam_search"
        
        print('\n~Validating~\n')

        example_input = []
        example_output = []
        predict_sequence = []
        predict_height = []
        avg_reward = []

        # put in test mode!
        model.eval()

        for batch_id, val_batch in enumerate(tqdm(validation_dataloader,
                disable=args['disable_progress_bar'])):
            bat = Variable(val_batch)

            if args['use_cuda']:
                bat = bat.cuda()

            R, probs, actions, action_idxs = model(bat)
            
            avg_reward.append(R[0].item())
            val_step += 1.

            if not args['disable_tensorboard']:
                log_value('val_avg_reward', R[0].item(), int(val_step))

            if val_step % int(args['log_step']) == 0:
                example_output = []
                example_input = []
                for idx, action in enumerate(actions):
                    # if task[0] == 'tsp':
                    example_output.append(action_idxs[idx][0].item())
                    # else:
                    # example_output.append(action[0].numpy())
                    example_input.append(bat[0, :, idx].numpy()) # 尝试item改numpy
                # print('Step: {}'.format(batch_id))
                # #print('Example test input: {}'.format(example_input))
                # print('Example test output: {}'.format(example_output))
                # print('Example test reward: {}'.format(R[0].item()))
                predict_sequence.append(example_output)
                predict_height.append(R[0].item())
            
                if args['plot_attention']:
                    probs = torch.cat(probs, 0)
                    plot_attention(example_input,
                            example_output, probs.data.cpu().numpy())
        print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
        print('Validation overall reward var: {}'.format(np.var(avg_reward)))
        with open('rewards_val.csv',"a+") as csvfile:
            csvfile.write(str(i)+' '+str(np.mean(avg_reward).tolist())+' '+str(np.var(avg_reward).tolist())+'\n')
        predict_sequence=np.array(predict_sequence)
        np.savetxt(os.path.join(save_dir, 'sequence-{}.csv'.format(i)),predict_sequence,fmt='%d')
        predict_height=np.array(predict_height)
        np.savetxt(os.path.join(save_dir, 'height-{}.csv'.format(i)),predict_height,fmt='%.05f')

        if args['is_train']:
            model.actor_net.decoder.decode_type = "stochastic"
            
            print('Saving model...')
        
            torch.save(model, os.path.join(save_dir, 'epoch-{}.pt'.format(i)))

            # If the task requires generating new data after each epoch, do that here!
    '''        if COP == 'tsp':
                training_dataset = tsp_task.TSPDataset(train=True, size=size,
                    num_samples=int(args['train_size']))
                training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
                    shuffle=True, num_workers=1)
            if COP == 'sort':
                train_fname, _ = sorting_task.create_dataset(
                    int(args['train_size']),
                    int(args['val_size']),
                    data_dir,
                    data_len=size)
                training_dataset = sorting_task.SortingDataset(train_fname)
                training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
                        shuffle=True, num_workers=1)'''
