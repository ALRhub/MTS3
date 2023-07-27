import sys
import os

sys.path.append('.')
from dataFolder.dataDpssmVar import metaData
import numpy as np
import json
import pickle
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import pandas as pd
from utils.dataProcess import normalize, denormalize


class metaExData(metaData):
    def __init__(self, data_cfg=None):
        super(metaExData, self).__init__(data_cfg)
        if data_cfg is None:
            raise Exception('Please specify a valid Confg for data')
        else:
            self.c = data_cfg
            self.c = data_cfg
        self._down_sample = self.c.downsample
        self.episode_length = self.c.episode_length  # window length for a single hiprssm instance
        self.num_episodes = self.c.num_episodes  # number of hiprssm instances

        self._trainPath =  get_original_cwd() + '/dataFolder/actuator_model_v2/data/export/train.pickle'
        self._valPath =  get_original_cwd() + '/dataFolder/actuator_model_v2/data/export/val.pickle'
        self._testPath =  get_original_cwd() + '/dataFolder/actuator_model_v2/data/export/test.pickle'



        self.train_windows, self.val_windows, self.test_windows = self._load_trajectories()
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}



    def get_statistics(self, data, dim, difference=False):
        if difference:
            data = (data[:, 1:, :dim] - data[:, :-1, :dim])
        reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], -1)) ### important since we have 4 dimension in Var style preprocessing
        data = reshape(data);
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std

    def _load_trajectories(self):
        # load the pickle file of trajectories
        print(self._trainPath,self._testPath)

        train_obs, train_act, train_target, train_task = self._loop_data("train")
        val_obs, val_act, val_target, val_task = self._loop_data("val")
        test_obs, test_act, test_target, test_task = self._loop_data("test")
        print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', train_obs.shape, test_obs.shape)

        train_data_window, test_data_window = self._pre_process(train_obs, train_act, train_target, train_task,
                                                                                 test_obs, test_act, test_target, test_task)
        _, val_data_window = self._pre_process(train_obs, train_act, train_target, train_task,
                                                                                 val_obs, val_act, val_target, val_task)

        return train_data_window, val_data_window, test_data_window

    def _loop_data(self,type='train'):
        if type == 'train':
            path = self._trainPath
        elif type == 'val':
            path = self._valPath
        elif type == 'test':
            path = self._testPath
        else:
            raise Exception('Please specify a valid type of data')
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        firstFlag = True
        i = 0
        for key in data_dict.keys():
            obs = np.array(data_dict[key])[:- 1, [1, 2]]
            act = np.array(data_dict[key])[:- 1, [3, 4]]
            # data['tasks'] = np.array(data_np['tasks'][:,:-1,:])
            next_obs = np.array(data_dict[key])[1:, [1, 2]]
            print('>>>>>>>>>>>>Processed Data Trajectories with shape<<<<<<<<<<<<<<<', obs.shape,
                  act.shape)

            ### Downsample
            obs = obs[::self._down_sample, :]
            act = act[::self._down_sample, :]
            next_obs = next_obs[::self._down_sample, :]
            num_seqs = self.c.num_seq_multiplier*int(obs.shape[0]/(self.num_episodes*self.episode_length))
            task_idx = i*np.ones((obs.shape[0]))
            obss = np.expand_dims(obs,0)
            acts = np.expand_dims(act,0)
            tasks = np.expand_dims(task_idx,0)
            next_obss = np.expand_dims(next_obs,0)
            print(tasks.shape)
            obs_batch, act_batch, target_batch, t_idx_batch = self._get_batch(obss,acts,next_obss,tasks,num_seqs)

            if firstFlag:
                full_obs = obs_batch
                full_act = act_batch
                full_target = target_batch
                full_task_idx = t_idx_batch
                firstFlag = False
            else:
                full_obs = np.concatenate((full_obs, obs_batch))
                full_act = np.concatenate((full_act, act_batch))
                full_target = np.concatenate((full_target, target_batch))
                full_task_idx = np.concatenate((full_task_idx, t_idx_batch))
                pass
            i = i+1
        return np.array(full_obs), np.array(full_act), np.array(full_target), np.array(full_task_idx)

    def _get_batch(self, obs, act, target, task_idx, num_seqs):
        # Takes multiple paths and splits them into windows based on random locations within a trajectory
        num_paths, len_path = obs.shape[:2]
        idx_path = np.random.randint(0, num_paths,
                                     size=num_seqs)  # task index, which gets mixed along the
        # process
        idx_batch = np.random.randint(0, len_path - self.num_episodes*self.episode_length, size=num_seqs)
        obs_batch = np.array([obs[ip,
                              ib:ib + self.num_episodes*self.episode_length, :]
                              for ip, ib in zip(idx_path, idx_batch)])
        obs_batch = np.transpose(np.array(np.split(obs_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
        act_batch = np.array([act[ip,
                              ib:ib + self.num_episodes*self.episode_length, :]
                              for ip, ib in zip(idx_path, idx_batch)])
        act_batch = np.transpose(np.array(np.split(act_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
        target_batch = np.array([target[ip,
                                 ib:ib + self.num_episodes*self.episode_length, :]
                                 for ip, ib in zip(idx_path, idx_batch)])
        target_batch = np.transpose(np.array(np.split(target_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
        t_idx_batch = np.array([task_idx[ip,
                                ib:ib + self.num_episodes*self.episode_length]
                                for ip, ib in zip(idx_path, idx_batch)])
        if len(t_idx_batch.shape) == 2:
            t_idx_batch = np.transpose(np.array(np.split(t_idx_batch, self.num_episodes,axis=1)),axes=[1,0,2])
        else:
            t_idx_batch = np.transpose(np.array(np.split(t_idx_batch, self.num_episodes, axis=1)), axes=[1, 0, 2,3])

        return np.array(obs_batch), np.array(act_batch), np.array(target_batch), np.array(t_idx_batch)



if __name__ == '__main__':
    dataFolder = os.getcwd() + '/dataFolder/MobileRobot/sin2/'
    # self._trajectoryPath = self._dataFolder + 'HalfCheetahEnv_6c2_cripple.pickle'
    trajectoryPath = dataFolder + 'ts_002_50x2000.npz'
    data = np.load(trajectoryPath)
    print(data.keys())
    print(np.sin(data['orn_euler']))
    print(np.cos(data['orn_euler']))
    print(data['orn_euler'])

