import os
# from PIL import Image
import random
import shutil

# import SimpleITK as sitk
import numpy  as np
import pandas as pd

from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader

##
import __init__

from loss     import get_loss
from optim    import get_optim, get_scheduler
from networks import get_net

from baseUtils.yamlConfig import get_config, dict2yaml
from baseUtils.kit        import mkdir, norm
from datasets.datasets    import landmarks


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Runner(object):
    def __init__(self, args):
        self.args  = args
        self.phase = self.args.phase

    def run(self):
        
        self.config()
        
        self.best_loss = self.train_loss = float('inf')
        if self.phase == 'train':
            self.train()
        else:
            self.validateTest(self.start_epoch)

    def config(self):
        self.get_opts()
        self.get_loader()
        self.get_model()

    def get_opts(self):
        self.opts     = get_config(self.args.config)

        self.run_name = self.opts['run_name'] if self.opts['run_name'] else self.opts['model'] 
        self.run_dir  = os.path.join(self.opts['run_dir'], self.run_name)
        mkdir(self.run_dir)
        dict2yaml("{rd}/config_{ph}.yaml".format(rd=self.run_dir, ph=self.phase), self.opts)
        shutil.copy(self.args.config, '{run_dir}/config_origin.yaml'.format(run_dir=self.run_dir))
        # switch cuda
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opts['cuda_devices']

        self.set_seed(self.opts['seed'])

    # 在get_opts()里被调用
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_loader(self):

        def _get(str_stage='train'):
            
            # 训练集需带数据增强的相关参数
            is_trans = self.opts['transform_params'] if str_stage == 'train' else False
            # is_trans  = self.opts['transform_params']
            # is_heatmap = self.opts['is_heatmap']
            
            # 读入datasets
            mydataset = landmarks(prefix=self.opts['prefix'], 
                                phase=str_stage, 
                                size=self.opts['size'],
                                sigma=self.opts['sigma'],
                                num_landmark=self.opts['landmarks_num'],
                                is_transform=is_trans,
                                use_background_channel=self.opts['use_background_channel'],
                                is_heatmap=self.opts['is_heatmap']
                                )

            # 迭代对象mydataloader
            loader_opts = self.opts['dataloader'][str_stage]
            mydataloader = DataLoader(mydataset, **loader_opts)

            setattr(self, str_stage + '_dataset', mydataset)
            setattr(self, str_stage + '_loader',  mydataloader)
        ##ondef

        if self.phase == 'train':
            _get('train')
            _get('validate')
        elif self.phase == 'test':
            _get('test')
        else:
            _get('validate')

    # get_model()调用
    def get_learner(self):
        # 损失和学习率优化的相关参数
        learn = self.opts['learning']
        self.loss     = get_loss(learn['loss'])(**learn[learn['loss']])
        self.val_loss = get_loss(learn['loss'])(**learn[learn['loss']])

        self.optim    = get_optim(learn['optim'])(self.model.parameters(), **learn[learn['optim']])
        
        
        if learn['use_scheduler']:
            self.scheduler = get_scheduler(learn['scheduler'])(
                self.optim, **learn[learn['scheduler']])
        else:
            self.scheduler = None

    # 记录使用的模型，并写到network_graph.txt
    def get_model(self):
        model_name  = self.opts['model']
        size        = self.opts['size']

        net_params = self.opts[model_name] if model_name in self.opts else dict()
        print("Loading model: {}".format(model_name))
        self.model = get_net(model_name)(**net_params)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        self.device = device
        self.model.to(device)

        ## 保存模型和超参数
        if os.path.isfile(self.opts['checkpoint']):
            print('loading checkpoint:', self.opts['checkpoint'])
            checkpoint       = torch.load(self.opts['checkpoint'])
            # print('checkpoint epoch:', checkpoint.keys())
            # self.start_epoch = checkpoint['epoch'] + 1
            self.start_epoch = 0
            self.model.load_state_dict(checkpoint['model_state_dict']) # 保存参数
            self.get_learner()
        else:
            self.start_epoch = 0
            self.get_learner()
        
        with open(os.path.join(self.run_dir, 'network_graph.txt'), 'w') as f:
            f.write(str(self.model))
    ##ondef

    # 训练
    def train(self):
        self.model.train()
        checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        mkdir(checkpoint_dir)
        pbar = tqdm(range(self.start_epoch, self.opts['epochs']))
        xs, ys = [], []
        self.lr_list = [] # len=batch_num
        loss_file = os.path.join(checkpoint_dir, 'train_val_loss.txt')
        endEpoch  = self.opts['epochs'] - 1
        save_freq = self.opts['save_freq']
        eval_freq = self.opts['eval_freq']
        """一个epoch一循环"""
        train_loss = []
        valid_loss = []
        for epoch in pbar:
            self.update_params(epoch, pbar) 
            # plot_2d(self.run_dir + '/learning_rate.png', list(range(len(self.lr_list))), self.lr_list, 'step', 'lr', 'lr-step')          
            if epoch % eval_freq == 0 or epoch == endEpoch:
                val_loss = self.validateTest(epoch)
                xs.append(epoch)
                ys.append(val_loss) # 存每个epoch后的val_loss
                # plot_2d(self.run_dir + '/loss.png', xs, ys,
                #         'epoch', 'loss', 'epoch-loss')
                data = {
                    'epoch': epoch,
                    # 'model_name': self.run_name,
                    'model_state_dict': self.model.state_dict(),
                    # 'optimizer': self.optim,
                    # 'scheduler': self.scheduler,
                }
               
                save_name = "{rn}_epoch{epoch:03d}_train{trainloss:.6f}_val{valloss:.6f}.pt".format(
                    rn=self.run_name, epoch=epoch, valloss=val_loss, trainloss=self.train_loss)
                with open(loss_file, 'a') as f:
                    f.write('{:03d},{:.6f},{:.6f}\n'.format(
                        epoch, self.train_loss, val_loss))       
                # 写入checkpoints目录下              
                if (save_freq != 0 and epoch % save_freq == 0) or epoch == endEpoch:
                    torch.save(data, os.path.join(checkpoint_dir, save_name))
                # 若同时满足best_loss则记录，并使用best_loss下的参数作为下一次train时的参数
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    dest = os.path.join(checkpoint_dir, 'best_' + save_name)
                    self.opts['checkpoint'] = dest
                    torch.save(data, dest)

            # print("Train loss: {:.3f}", self.train_loss)
            # print("Val loss: {:.3f}", val_loss)
            train_loss.append(self.train_loss)
            valid_loss.append(val_loss)

        ##save loss to excel
        loss_file = os.path.join(self.run_dir, 'loss.xlsx')
        df_loss = pd.DataFrame({'train_loss': train_loss, 'valid_loss': valid_loss})
        df_loss.to_excel(loss_file, index=False)
    ##ondef

    # 验证
    def validateTest(self, epoch=None):
        self.model.eval()  # important，把BatchNormalization和DropOut固定
        if epoch is None:
            epoch = self.start_epoch
        prefix = self.run_dir + '/results/' + self.phase + \
                '_epoch{epoch:03d}'.format(epoch=epoch)
        loss_dir = self.run_dir + '/results/loss'
        mkdir(loss_dir)
        mkdir(prefix)
        s = 'validate' if self.phase == 'train' else self.phase
        cur_loader = getattr(self, '{}_loader'.format(s))
        val_loss   = 0
        allep = self.opts['epochs']
        
        batch_num = len(cur_loader)
        batch_size = self.opts['dataloader']['validate']['batch_size']

        pbar = tqdm(enumerate(cur_loader))  # is read in func save_data
        name_loss_dic = {}
        for i, data_dic in pbar:
            for k in {'input', 'gt'}:
                if k in data_dic:
                    data_dic[k] = torch.autograd.Variable(
                        data_dic[k]).to(self.device)
            with torch.no_grad(): # 保证各个节点的参数不会被更新
                output_pred = self.model(data_dic['input'])

                # if not self.opts['is_heatmap']:
                #     output_pred = output_pred.reshape(-1, self.opts['landmarks_num'], 2)
                
                data_dic['output'] = output_pred



            # if epoch + 1 >= allep or self.phase != 'train':
            #     save_data(data_dic)
            if 'gt' in data_dic:
                if data_dic['output'].shape != data_dic['gt'].shape:
                    print(data_dic['path'])
                    exit()
                loss = self.val_loss(data_dic['output'], data_dic['gt'])  # TODO
                if 'rec_image' in data_dic:
                    # 用的是均方损失
                    loss += get_loss('l2')(**self.opts.learning.l2)(data_dic['input'], data_dic['rec_image'])

                """在终端显示验证信息，这个地方为什么会连续输出两行一摸一样的"""
                
                pbar.set_description("[{curPhase}] epoch:{ep:>3d}/{allep:<3d}, batch:{num:>6d}/{batch_num:<6d}, train:{ls:.6f}, vali:{val_loss:.6f}".format(ep=epoch, allep=allep, num=(i + 1), batch_num=batch_num, val_loss=loss.item()/batch_size, ls=self.train_loss, curPhase=s.ljust(8)))
                name_loss_dic['_'.join(data_dic['name'])] = loss.item()
        # # 
        if 'gt' in data_dic:
            mean     = np.mean(list(name_loss_dic.values()))
            val_loss = mean
            path = os.path.join(
                loss_dir, 'epoch_{:03d}_loss_{:.6f}.txt'.format(epoch, mean))
            with open(path, 'w') as f:
                for k, v in name_loss_dic.items():
                    # 把验证集每个图片跑出的误差记录下来
                    f.write('{:.6f} {}\n'.format(v, k))
        # print(val_loss)       
        return val_loss
    ##ondef

    # 参数更新
    def update_params(self, epoch, pbar):
        # to try: harmonic mean
        self.model.train()  # important
        self.train_loss = 0  # sum of different datasets' arithmetic mean
        allep = self.opts['epochs']
        use_scheduler = self.opts['learning']['use_scheduler']
        loader     = self.train_loader
        batch_num  = len(self.train_loader)
        batch_size = self.opts['dataloader']['train']['batch_size']
        cur_loss   = 0 # 当前epoch的每个batch的loss累加

        # 一个batch的训练，data_dic['name','inpt','gt']为batch的字典
        for i, data_dic in enumerate(loader):
            for k in {'input', 'gt'}:
                data_dic[k] = torch.autograd.Variable(data_dic[k]).to(self.device)

            """进入模型训练"""
            output_pred = self.model(data_dic['input'])

            # if not self.opts['is_heatmap']:
            #     output_pred = output_pred.reshape(-1, self.opts['landmarks_num'], 2)

            data_dic['output'] = output_pred
            
            # 清零方便下一次调用
            self.optim.zero_grad()


            loss = self.loss(data_dic['output'], data_dic['gt'])  # TODO

            # import sys
            # sys.exit(0)

            if 'rec_image' in data_dic:
                loss += get_loss('l2')(**self.opts.learning.l2)(data_dic['input'], data_dic['rec_image'])
            
            cur_loss += loss.item()

            # 反向传播
            loss.backward()
            self.lr_list.append(self.optim.param_groups[0]['lr'])

            # 学习率的调整应该放在optimizer更新之后
            self.optim.step()
            if use_scheduler:
                self.scheduler.step()  # behind optim.step()
            
            # show in terminal
            pbar.set_description("[train] epoch:{ep:>3d}/{allep:<3d}, batch:{num:>6d}/{batch_num:<6d}, train:{ls:.6f}, best:{val_loss:.6f}".format(
                 ep=epoch, allep=allep, num=i + 1, batch_num=batch_num, ls=loss.item(), val_loss=self.best_loss))
            
            # pbar.set_description("[train] epoch:{ep:>3d}/{allep:<3d}, batch:{num:>6d}/{batch_num:<6d}, train:{ls:.6f}, best:{val_loss:.6f}".format(
            #      ep=epoch, allep=allep, num=i + 1, batch_num=batch_num, ls=loss.item()/batch_size, val_loss=self.best_loss))
        
        self.train_loss = cur_loss / (len(loader)*batch_size)

        return self.train_loss
    ##ondef