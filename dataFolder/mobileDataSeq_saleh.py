import sys
import os

sys.path.append('.')
from torch.utils.data import Dataset
import numpy as np
import pickle
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import pandas as pd
import configparser
import os
import collections
import yaml
import torch
import time


class metaMobileData(Dataset):
    def __init__(self, data_cfg=None):
        if data_cfg is None:
            raise Exception('Please specify a valid Config for data')
        else:
            self.c = data_cfg.data_reader
            self.l = data_cfg.learn   #ADDED BY SALEH
            self.d = data_cfg.data_reader

        self.dataFolder = os.path.join(os.path.dirname(get_original_cwd()) , 'all_data' ,'workload' ) #saleh added to have only 1 data folder for all projects// #os.path.join(get_original_cwd() , 'data' ,'workload' )
        self.dataFolder = os.path.join(os.path.dirname(os.getcwd()), 'all_data', 'workload')
        self._trajectoryPath = self.dataFolder + '/allArrays_1to50_sz74_19clustered.pickle'
        print("data_path:",self._trajectoryPath)

        self._save_windows = self.c.save
        self._load_windows = self.c.load
        self.dim = self.c.dim #(1)
        #print("line 40 mobileDataSeq.py self.dim=",self.dim)
        self.trajPerTask = self.c.trajPerTask ## number of time series corresponding to particular load of robot     @10ms (#flows, 60000 ,50,1 )

        self.tar_type = self.c.tar_type

        self._split = OmegaConf.to_container(self.c.split)
        self._shuffle_split = self.c.shuffle_split

        self.meta_batch_size = self.c.meta_batch_size # number of chunks
        self.batch_size = self.c.batch_size  # window length/2  # time series window length : 50
        self.normalization = None
        self.filename = self.c.file_name
        self.standardize = self.c.standardize
        self.standardize_dist = self.c.standardize_dist # saleh_added

        self.loss = self.l.loss  # if 'cross_entropy' then we should not normalize labels #ADDED BY SALEH
        self.features_in  = self.l.features_in
        self.features_out = self.l.features_out
        self.mb = self.d.meta_batch_size

        self.train_windows, self.test_windows = None , None
        #self.train_windows, self.test_windows = self.load_data()  # we load in the main file
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}
        #print("MobileDataSeq.py  line ~53_Saleh_added: self.train_windows, self.test_windows = self.load_data()  done!")




    def normalize(self, data, mean, std):
        dim = data.shape[-1]  #1
        return (data - mean[:self.dim]) / (std[:self.dim] + 1e-10)

    def denormalize(self, data, mean, std):
        dim = data.shape[-1] #1
        return data * (std[:self.dim] + 1e-10) + mean[:self.dim]

    def get_statistics(self, data, dim, difference=False):
        #print("saleh_added: checking function get_statistics: mobileDataSeq.py")
        #print("difference = ",difference)
        if difference:
            data = (data[:, 1:, :self.dim] - data[:, :-1, :self.dim])
            #print("data.shape=",data.shape)
        reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
        data = reshape(data);
        #print("saleh_added  get_statistics, data.shape = ",data.shape)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        #print("saleh_added: mean=",mean, "  std=",std)
        return mean, std

    def load_data(self):
        # load the pickle file of trajectories
        # if self._load_windows is not None: #False
        #     train_data_window = pickle.load(open(self._dataFolder + self.filename + '_train.pickle', 'rb'))
        #     test_data_window = pickle.load(open(self._dataFolder + self.filename + '_test.pickle', 'rb'))
        #     self.normalization = train_data_window['normalization']
        #     print('>>>>>>>>>>>>Loaded Saved Windows with shape<<<<<<<<<<<<<<<', train_data_window['obs'].shape)

    #else:  #True
        print('generating trajectories by sampling from flows .....')
        print("reading raw array:")
        #print(self._trajectoryPath)
        with open(self._trajectoryPath, 'rb') as f:
            data_np = pickle.load(f)
            data_np=data_np[:,:800000,:] #100000*1.5ms/60000 = 2.5 min  => 400000 sample~10min
            print(data_np.shape) #[felow_ids,time,51(hist)]
        print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', data_np.shape)  #>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<< (50, 1750, 4)
                                                                                                                                         # fs= @10ms: data~ (#flows,12000,50 samples)

        # collect obs, act, next states
        data = {'observations':[], 'actions':[], 'next_observations':[]}
        print("Saleh_added:  data = {'observations':[], 'actions':[], 'next_observations':[]}     MobileDataSeq.py  line ~85")

        # dim = np.r_[0:15, 30:41]
        data['observations'] = (data_np[:,:-1,:])
        data['actions'] = np.zeros_like(data['observations'])

        #print("saleh_added mobileDataSeq.py  line91,  data['action'] = ", data['actions']) #saleh_added

        data['grad'] = np.squeeze(data['observations'][:,:,0]+1)  #only for vis can be replaced with cluster   #+1 I added for no reaseon/ just to make a difference between grad & obs
        print('saleh_added line 101 mobileseqdata.py',data['grad'].shape)
        #data['grad'] = np.squeeze(data['observations']+1)  #only for vis can be replaced with cluster   #+1 I added for no reaseon/ just to make a difference between grad & obs
        data['next_observations'] = (data_np[:,1:,:])
        obs = data['observations']
        act = data['actions']
        next_obs = data['next_observations']
        grad = data["grad"]
        print('>>>>>>>>>>>>Processed Data Trajectories with shape<<<<<<<<<<<<<<<', obs.shape,act.shape)


        # train test split  #modified by saleh
        train_obs, train_act, train_next_obs, train_grad, test_obs, test_act, test_next_obs, test_grad = self.train_test_split(obs, act, next_obs,grad)

        #   here label is the last feature which is number of packets in that interval. it can also be defined differently e.g label is the #packets in the next interval.
        train_packets      = train_obs[...,-1:]               #added by saleh
        train_next_packets = train_next_obs[..., -1:]         #added by saleh
        train_packets_delta = train_next_packets - train_packets  #added by saleh

        test_packets      = test_obs[...,-1:]      #added by saleh
        test_next_packets = test_next_obs[...,-1:] #added by saleh
        test_packets_delta = test_next_packets - test_packets



        # Note: you do not need to one-hot encode the labels. The loss functions expect an integer value with the corresponding class.
        #https://wandb.ai/capecape/classification-techniques/reports/Classification-Loss-Functions-Comparing-SoftMax-Cross-Entropy-and-More--VmlldzoxODEwNTM5


        #train_obs_delta = train_next_obs - train_obs  #irrelevant in regression setting & notation
        #test_obs_delta = test_next_obs - test_obs     #irrelevant in regression setting & notation




        mean_packets_diff, std_packets_diff = self.get_statistics(train_packets_delta, self.dim)#saleh_added

        mean_obs, std_obs = self.get_statistics(train_obs, self.dim)  # obs means concate([dist arrival,#packets]) ---> 25 dimensional
        mean_act, std_act = self.get_statistics(train_act, 2 * self.dim)  # would return zero
        mean_packets, std_packets = self.get_statistics(train_packets, self.dim)


        self.normalization = dict()
        self.normalization['observations']  = [mean_obs, std_obs] #mean and std of actual observations
        self.normalization['actions']       = [mean_act, std_act]
        #self.normalization['obs_diff']      = [mean_obs_diff, std_obs_diff] # irrelevant in regression notation & setting
        self.normalization['packets']         = [mean_packets, std_packets]
        self.normalization['packets_diff']    = [mean_packets_diff, std_packets_diff]

        print("************************************************************************************************************************")
        print("tar_type:{}, features_in:{}, features_out:{}, loss:{}".format(self.tar_type, self.features_in , self.features_out, self.loss))
        print("************************************************************************************************************************")

        if(self.features_in == 'all' and self.tar_type == 'observations'):
            self.normalization['input'] = [mean_obs, std_obs]
            if(self.features_out == 'all'):
                self.normalization['target'] = [mean_obs, std_obs]
            elif(self.features_out == 'packets'):
                self.normalization['target'] = [mean_packets, std_packets]
            else:
                raise NotImplementedError

        elif(self.features_in == 'packets'and self.tar_type=='observations'):
            if(self.features_out == 'all'):
                    raise NotImplementedError
            self.normalization['input'] = [mean_packets, std_packets]
            self.normalization['target']= [mean_packets, std_packets]

        elif(self.features_in == 'packets'and self.tar_type=='delta'):
            if(self.features_out == 'all'):
                    raise NotImplementedError
            self.normalization['input']  = [mean_packets_diff, std_packets_diff]
            self.normalization['target'] = [mean_packets_diff, std_packets_diff]
        else:
            raise NotImplementedError

        # delta (if True) would only applies on the target and not the input
        # Standardize
        if (self.standardize or self.standardize_dist):  # True
            print(">>>>>>>>>>>>>>>>>>>>>>>>>Standardizing The Data<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            if(self.standardize and not(self.standardize_dist)): #only packet
                self.train_obs  = self.normalize(train_obs[:,:,-1:], self.normalization["packets"][0],self.normalization["packets"][1])
                self.train_acts = self.normalize(train_act[:,:,-1:], self.normalization["packets"][0]     ,self.normalization["packets"][1])

                self.test_obs   = self.normalize(test_obs[:,:,-1:], self.normalization["packets"][0],self.normalization["packets"][1])
                self.test_acts  = self.normalize(test_act[:,:,-1:], self.normalization["packets"][0],self.normalization["packets"][1])


            if (self.tar_type == 'delta' and self.features_out=='packets') : # usually True but usualy not in clustering/classification schema !  #
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Training On differences <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                # in the next 6 lines target targets are #number of packest (for the next interval) or 1st_diff of number of packets
                self.train_target = self.normalize(train_packets_delta, self.normalization["packets_diff"][0],self.normalization["packets_diff"][1]) #Prediction target
                self.test_target  = self.normalize(test_packets_delta , self.normalization["packets_diff"][0],self.normalization["packets_diff"][1]) #Prediction target
                #self.normalization['targets'] = [mean_label_diff, std_label_diff]

            elif (self.tar_type == 'observations' and self.features_out=='packets'):
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Training On Next Observation_packet (not differences) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("self.tar_type:",self.tar_type)
                if(self.loss == 'cross_entropy'):  # when it is not cross_entropy we should normalize the labels
                    self.train_target = train_next_packets
                    self.test_target  = test_next_packets
                elif (self.loss in (['mse','mae'])): # when it is not cross_entropy we should normalize the labels
                    self.train_target= self.normalize(train_next_packets     , self.normalization["packets"][0] ,self.normalization["packets"][1])       #Prediction target
                    self.test_target = self.normalize(test_next_packets      , self.normalization["packets"][0] ,self.normalization["packets"][1])       #Prediction target
                else:
                    print('invalid combination of params')
                    raise NotImplementedError
            elif (self.tar_type == 'observations' and self.features_out == 'all' and self.loss == 'mse'):

                self.train_target = self.normalize(train_next_obs,self.normalization["observations"][0],self.normalization["observations"][1])
                self.test_target  = self.normalize(test_next_obs ,self.normalization["observations"][0],self.normalization["observations"][1])

            elif (self.tar_type == 'observations' and self.features_out == 'all' and self.loss == 'kl_rmse'): # KL div on arrival times & cross entropy on #num_packets
                print("normalization is Done as follow:")
                print("Not difference|| features_in = {} || features_out ={} || loss = {}".format(self.features_in,self.features_out,self.loss))
                print("train_obs[:]||test_obs[:]|| ----->   input(X)(not target y) only packets are normalized")
                print("train_next_obs([:-1])||test_next_obs[:-1]|| -----> target Y Distribution Not normalized")
                print("train_next_obs([-1:])||test_next_obs[-1:]|| -----> target Y Packets are Normalized with self.normalization[packets]")
                print("summary: both in test and train (obviously), distribution are not normalized ( in both input:X and label:Y)/// packets are normalized(both input:X and label:Y) ")
                print("====================================================================================================")

                # self.train_obs  = self.normalize(train_obs, self.normalization["observations"][0],self.normalization["observations"][1])
                # self.train_acts = self.normalize(train_act, self.normalization["actions"][0]     ,self.normalization["actions"][1])
                #
                # self.test_obs   = self.normalize(test_obs, self.normalization["observations"][0],self.normalization["observations"][1])
                # self.test_acts  = self.normalize(test_act, self.normalization["actions"][0]     ,self.normalization["actions"][1])


                self.train_target  = train_next_obs
                self.test_target   = test_next_obs

                self.train_target[:,:,-1:]  = self.normalize(train_next_packets     , self.normalization["packets"][0] ,self.normalization["packets"][1])       #Prediction target in trainin
                self.test_target [:,:,-1:]  = self.normalize(test_next_packets      , self.normalization["packets"][0] ,self.normalization["packets"][1])       #Prediction target in testing



            else:
                print('invalid combination of params')
                raise NotImplementedError

        else: #Not standardized version ---> False (almost always)
            self.train_obs = train_obs
            self.train_acts = train_act
            self.test_obs = test_obs
            self.test_acts = test_act


            if self.tar_type == 'delta' and self.features_out=='packets':
                self.train_target = train_packets_delta
                self.test_target  = test_packets_delta
            if self.tar_type == 'delta' and self.features_out=='all':
                raise NotImplementedError
            if self.tar_type == 'observations' and self.features_out=='all':
                self.train_target = train_next_obs
                self.test_target = test_next_obs
            else:
                print('invalid combination of params')
                raise NotImplementedError
        self.train_task_idx = train_grad   # train_grad , test_grad hamun grad hastand ke split shode
        self.test_task_idx = test_grad   #idx to  cluster number only for vis purpose

        # Get random windows ---> the next 2 lines is the output of this script
        print("generating tran&test windows")
        train_data_window = self._get_batch(train=True)
        test_data_window = self._get_batch(train=False)


        if self._save_windows is not None:
            # t1 = time.time()
            # pickle.dump(train_data_window, open(self._dataFolder + self.filename + '_train.pickle', 'wb'))
            # pickle.dump(test_data_window, open(self._dataFolder + self.filename + '_test.pickle', 'wb'))
            # t2 = time.time()

            txt_train = 'train_data_window_' +str(self.mb)+ '.npz'
            txt_test = 'test_data_window_' + str(self.mb) + '.npz'
            print("saving .... ",txt_train)
            np.savez(txt_train, **train_data_window)
            np.savez( txt_test, **test_data_window)
            # t3 = time.time()
            # print('efficiency with npz:',(t2-t1)/(t3-t2))

        #print("saleh_added line 207 ,     isinstance(self._shuffle_split, float)==",isinstance(self._shuffle_split, float) )-->False
        if isinstance(self._shuffle_split, float): #False
            dataset_size = train_data_window['obs'].shape[0]
            indices = np.arange(dataset_size)
            print(indices, dataset_size)
            np.random.shuffle(indices)
            print(indices)
            split_idx = int(dataset_size * self._shuffle_split)
            idx_train = indices[:split_idx]
            idx_test = indices[split_idx:]
            train_set = {'obs': [], 'act': [], 'target': [], 'task_index': [], 'normalization': []}
            test_set = {'obs': [], 'act': [], 'target': [], 'task_index': [], 'normalization': []}


            train_set['obs'] = train_data_window['obs'][idx_train,:,:3];
            test_set['obs'] = train_data_window['obs'][idx_test,:,:3];
            train_set['act'] = train_data_window['act'][idx_train];
            test_set['act'] = train_data_window['act'][idx_test];

            print("saleh_added: line 226 mobileDataSeq.py")
            print(train_set['act'],test_set['act'])
            train_set['act'] =  np.zeros_like(train_set['act'])#saleh_added
            test_set['act']  =  np.zeros_like(test_set['act'])  #saleh_added

            train_set['target'] = train_data_window['target'][idx_train,:,:3];
            test_set['target'] = train_data_window['target'][idx_test,:,:3];
            train_set['task_index'] = train_data_window['task_index'][idx_train,:,3];
            test_set['task_index'] = train_data_window['task_index'][idx_test,:,3];
            train_set['normalization'] = self.normalization;
            test_set['normalization'] = self.normalization
            print('Train Test Split Ratio', self._shuffle_split)
            return train_set, test_set

        return train_data_window, test_data_window



    def _get_batch(self, train, percentage_imputation=0.0): # 3000 random point between 0to50
        # Takes multiple paths and splits them into windows based on random locations within a trajectory
        golden_value = self.normalize(np.zeros(25), self.normalization["observations"][0], self.normalization["observations"][1])[-1]
        zeros_cnt=0

        obs_batch   = []
        act_batch   = []
        label_batch = []
        t_idx_batch = []
        if train:
            print("saleh_added: line 278 mobileDataSeq.py  _get_batch: ,train_flag = ", train)
            num_paths, len_path = self.train_obs.shape[:2]
            idx_path = np.random.randint(0, num_paths,size=self.meta_batch_size)  # task index, which gets mixed along the  ## would select flow numbers,e.g 3500 times from 1st flow to last tarin flow
            idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size) # would select some middle points from time-steps [0:10000], 3500 times

            for ip, ib in zip(idx_path, idx_batch):

                #remove all zeros from train// later remove 90% not all of them
                #print(self.train_obs[ip,ib - self.batch_size:ib + self.batch_size, -1])
                if( (self.train_obs[ip,ib - self.batch_size:ib + self.batch_size, -1]==golden_value).all() ): #golden_value = normalized zeros
                    #print('flow_train:',ip ,"train_idx=", self._split[0][ip]  ,' t=',ib - self.batch_size,':',ib + self.batch_size, ' are all zero')
                    zeros_cnt = zeros_cnt + 1
                    a = np.random.uniform(low=0, high=1)
                    if(a>0.25): # take 20% of zeros vector
                        #print('discarded')
                        continue
                    #else:
                        #print("Not discarded")

                obs_batch.append(self.train_obs[ip,ib - self.batch_size:ib + self.batch_size, :] ) #if not all zeros
                #[3500(meta_batch),150(2*k),25(#features)]

                act_batch.append(self.train_acts[ip,ib - self.batch_size:ib + self.batch_size, :])

                #print("act_batch:",act_batch) #saleh_added  all 0 checked
                #act_batch=np.zeros_like(act_batch)
                label_batch.append(self.train_target[ip,ib - self.batch_size:ib + self.batch_size, :])
                t_idx_batch.append(self.train_task_idx[ip,ib - self.batch_size:ib + self.batch_size])
            print('train_zeros =',zeros_cnt)
        else: #test_data
            print("saleh_added: line 260 mobileDataSeq.py, _get_batch: train_flag = ", train)
            num_paths, len_path = self.test_obs.shape[:2]
            idx_path = np.random.randint(0, num_paths, size=self.meta_batch_size)
            idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size)

            for ip, ib in zip(idx_path, idx_batch):

                #remove all zeros from test// comment it later
                #print(self.test_obs[ip,ib - self.batch_size:ib + self.batch_size, -1])
                # if( (self.test_obs[ip,ib - self.batch_size:ib + self.batch_size, -1]==golden_value).all() ):
                #     print('flow_test:',ip ,"test_idx=", self._split[1][ip] ,' t=',ib - self.batch_size,':',ib + self.batch_size, ' are all zero')
                #     zeros_cnt=zeros_cnt+1
                #     #print(self.test_obs[ip,ib - self.batch_size:ib + self.batch_size, :])
                #     continue


                obs_batch.append(self.test_obs[ip,ib - self.batch_size:ib + self.batch_size, :])
                act_batch.append(self.test_acts[ip,ib - self.batch_size:ib + self.batch_size, :])


                #print("act_batch:",act_batch) #saleh_added # all 0 checked
                #act_batch=np.zeros_like(act_batch)

                label_batch.append(  self.test_target[ip,ib - self.batch_size:ib + self.batch_size, :])
                t_idx_batch.append(  self.test_task_idx[ip,ib - self.batch_size:ib + self.batch_size])
            #t_idx_batch   for visualization purpose
            print('test_zeros =', zeros_cnt)
        obs_batch   = np.stack(obs_batch,   axis=0)
        act_batch   = np.stack(act_batch,   axis=0)
        label_batch = np.stack(label_batch, axis=0)
        t_idx_batch = np.stack(t_idx_batch, axis=0)

        rs = np.random.RandomState(seed=42)
        obs_valid_batch = rs.rand(obs_batch.shape[0], obs_batch.shape[1], 1) < 1 - percentage_imputation
        obs_valid_batch[:, :5] = True

        #If delta is True ---> 'target' in the next line is based on difference
        data_windows = {'obs': obs_batch, 'act': act_batch, 'target': label_batch, 'obs_valid': obs_valid_batch,
                        'normalization': self.normalization,
                        'task_index': np.mean(t_idx_batch,-1)}  ###CLEVER TRICK %trajPerTask
        # TODO for the target(second half) initialize few things to True
        return data_windows


    def _get_batch_old(self, train, percentage_imputation=0.0): # 3000 random point between 0to50
        # Takes multiple paths and splits them into windows based on random locations within a trajectory
        if train:
            num_paths, len_path = self.train_obs.shape[:2]
            idx_path = np.random.randint(0, num_paths,
                                         size=self.meta_batch_size)  # task index, which gets mixed along the  ## would select flow numbers,e.g 3500 times from 1st flow to last tarin flow
            # process
            idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size) # would select some middle points from time-steps [0:10000], 3500 times



            obs_batch = np.array([self.train_obs[ip,
                                  ib - self.batch_size:ib + self.batch_size, :]
                                  for ip, ib in zip(idx_path, idx_batch) ]) #if not all zeros


            act_batch = np.array([self.train_acts[ip,
                                  ib - self.batch_size:ib + self.batch_size, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            print("saleh_added: line 238 mobileDataSeq.py  _get_batch: ,train_flag = ",train)
            #print("act_batch:",act_batch) #saleh_added  all 0 checked
            act_batch=np.zeros_like(act_batch)
            label_batch = np.array([self.train_target[ip,
                                     ib - self.batch_size:ib + self.batch_size, :]
                                     for ip, ib in zip(idx_path, idx_batch)])
            t_idx_batch = np.array([self.train_task_idx[ip,
                                    ib - self.batch_size:ib + self.batch_size]
                                    for ip, ib in zip(idx_path, idx_batch)])

        else: #test_data
            num_paths, len_path = self.test_obs.shape[:2]
            idx_path = np.random.randint(0, num_paths, size=self.meta_batch_size)
            idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size)

            obs_batch = np.array([self.test_obs[ip,
                                  ib - self.batch_size:ib + self.batch_size, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            act_batch = np.array([self.test_acts[ip,
                                  ib - self.batch_size:ib + self.batch_size, :]
                                  for ip, ib in zip(idx_path, idx_batch)])

            print("saleh_added: line 260 mobileDataSeq.py, _get_batch: train_flag = ",train)
            #print("act_batch:",act_batch) #saleh_added # all 0 checked
            act_batch=np.zeros_like(act_batch)

            label_batch = np.array([self.test_target[ip,
                                     ib - self.batch_size:ib + self.batch_size, :]
                                     for ip, ib in zip(idx_path, idx_batch)])
            t_idx_batch = np.array([self.test_task_idx[ip,
                                  ib - self.batch_size:ib + self.batch_size]
                                  for ip, ib in zip(idx_path, idx_batch)])
            #t_idx_batch   for visualization purpose

        rs = np.random.RandomState(seed=42)
        obs_valid_batch = rs.rand(obs_batch.shape[0], obs_batch.shape[1], 1) < 1 - percentage_imputation
        obs_valid_batch[:, :5] = True

        #If delta is True ---> 'target' in the next line is based on difference
        data_windows = {'obs': obs_batch, 'act': act_batch, 'target': label_batch, 'obs_valid': obs_valid_batch,
                        'normalization': self.normalization,
                        'task_index': np.mean(t_idx_batch,-1)}  ###CLEVER TRICK %trajPerTask
        # TODO for the target(second half) initialize few things to True

        return data_windows

    def train_test_split(self, obs, act, delta, grad):
        print(obs.shape[0],act.shape[0],delta.shape[0])
        assert obs.shape[0] == act.shape[0] == delta.shape[0]
        assert isinstance(self._split, list) or isinstance(self._split, float)
        episodes = obs.shape[0]

        assert len(self._split) == 2
        idx_train = self._split[0]
        idx_test = self._split[1]
        print('Training Indices:', idx_train, 'Testing Indices:', idx_test)

        # idx_test = [8,9]
        # print(len(idx_train) , idx_train)
        # print(len(idx_test)  , idx_test)
        # print(episodes)
        assert len(idx_train) + len(idx_test) <= episodes

        return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], grad[idx_train, :], \
               obs[idx_test, :], act[idx_test, :], delta[idx_test, :], grad[idx_test, :]

###### some temp ######

# def get_config_dict():
#     if not hasattr(get_config_dict, 'config_dict'):
#         get_config_dict.config_dict = dict(config.items('SECTION_NAME'))
#     return get_config_dict.config_dict
#
#
# def get_config_section():
#     if not hasattr(get_config_section, 'section_dict'):
#         print('1')
#         get_config_section.section_dict = collections.defaultdict()
#
#         for section in config.sections():
#             print('1')
#             get_config_section.section_dict[section] = dict(config.items(section))
#
#     return get_config_section.section_dict

def generate_mobile_robot_data_set(data, dim): #dim is useless. it was used in some older project in order not to use all features. it was set to 111 which is much bigger than our feature size
    #print("saleh_add: line 28 mobile_robot_hiprssm.py    train_target is only the #packet I.e [:,:,-1]")
    train_windows, test_windows = data.train_windows, data.test_windows
    #print('saleh_add: line 28 mobile_robot_hiprssm.py    dim=',dim)
    # train_targets = train_windows['target'][:,:,:dim] #remove dim --> : dim=111 but actual size =(3000,300,9) #or [learning_batchsize, data_reader_batch_size,features]
    # test_targets = test_windows['target'][:,:,:dim]
    #print("saleh_added line 33 mobile_robot_hiprssm.py  set the target (both in train&test) to the last feature of data")
    train_targets = train_windows['target'][:,:,-1:]
    test_targets = test_windows['target'][:,:,-1:]
    #print("saleh_add: line 36 mobile_robot_hiprssm.py")




    train_obs = train_windows['obs'][:,:,:]
    test_obs = test_windows['obs'][:,:,:]

    train_task_idx = train_windows['task_index'] # visualization
    test_task_idx = test_windows['task_index']

    train_act = train_windows['act'][:,:,:dim] #all 0 checked
    test_act = test_windows['act'][:,:,:dim] #all 0 checked

    print("train_targets.shape",train_targets.shape)
    print("test_targets.shape",test_targets.shape)
    print("mobile_robot_hiprssm.py line 50   data.normalization = ",data.normalization)
    #print(test_act.shape, train_act.shape) #(batch_size,2*data_reader_batch_size,#num_features) e.g (1000,2*75,10)

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(train_task_idx).float(),\
           torch.from_numpy(test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(test_targets).float(), torch.from_numpy(test_task_idx).float()



# if __name__ == '__main__':
#
#
#     config = configparser.RawConfigParser()
#     conf_dir = os.path.join( os.getcwd(),"experiments","mobileRobot" ,"conf","model","default_copy.yaml")
#     with open(conf_dir, "r") as f:
#         config = yaml.safe_load(f)
#     #config.read(conf_dir)
#     #print(config) # nested dict
#
#
#     dataFolder = os.path.join(os.path.dirname(os.getcwd()), 'all_data',
#                                     'workload')  # saleh added to have only 1 data folder for all projects// #os.path.join(get_original_cwd() , 'data' ,'workload' )
#     trajectoryPath = dataFolder + '/allArrays_1to50.pickle'
#     print("data_path:", trajectoryPath)
#
#     with open(trajectoryPath, 'rb') as f:
#         data_np = pickle.load(f)
#         data_np = data_np[:, :100000, :]  # N excluded ---> gradient problem
#         print(data_np.shape)  # [felow_ids,time,51(hist)]
#     print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', data_np.shape)
#
#     tar_type = config["data_reader"]["tar_type"]  # 'delta' - if to train on differences to current states
#     # 'next_state' - if to trian directly on the  next states
#
#     data = metaMobileData(config["data_reader"])
#
#     train_obs, train_act, train_targets, train_task_idx, test_obs, test_act, test_targets, test_task_idx = generate_mobile_robot_data_set(data, config["data_reader"]["dim"])
#     act_dim = train_act.shape[-1]

