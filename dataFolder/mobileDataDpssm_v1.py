import sys
import os

sys.path.append('.')
from dataFolder.dataDpssm import metaData
import numpy as np
import pickle
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import pandas as pd
from utils.dataProcess import normalize, denormalize
from matplotlib import pyplot as plt


class metaMobileData(metaData):
    def __init__(self, data_cfg=None):
        super(metaMobileData, self).__init__(data_cfg)
        if data_cfg is None:
            raise Exception('Please specify a valid Confg for data')
        else:
            self.c = data_cfg
            self.c = data_cfg
        if self.c.terrain=='sin2':
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sin2/'
            if self.c.frequency == '500':
                self._trajectoryPath = self._dataFolder + 'ts_002_50x2000_w_grad.npz'
            if self.c.frequency == '240':
                self._trajectoryPath = self._dataFolder + 'ts_def_50x1000_w_grad.npz'
        elif self.c.terrain=='sinMix2':
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sinMix2/'
            if self.c.frequency == '500':
                self._trajectoryPath = self._dataFolder + 'ts_def_50x1000_w_grad.npz'
            if self.c.frequency == '240':
                self._trajectoryPath = self._dataFolder + 'ts_def_50x1000_w_grad.npz'
        elif self.c.terrain=='both240':
            ###get Sine first
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sin2/'
            self._trajectoryPath1 = self._dataFolder + 'ts_def_50x1000_w_grad.npz'
            ###get SinMix Next
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sinMix2/'
            self._trajectoryPath2 = self._dataFolder + 'ts_def_50x1000_w_grad.npz'
        elif self.c.terrain=='sinLong2':
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sinLong2/'
            self._trajectoryPath = self._dataFolder + 'sin_1_ts_0.002_50x6000_w_grad.npz'
        elif self.c.terrain=='sinMixLong':
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sinMixLong/'
            self._trajectoryPath = self._dataFolder + 'sin_mx3_ts_def_50x4000_w_grad.npz'
        elif self.c.terrain == 'sinLong':
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sinLong/'
            self._trajectoryPath1 = self._dataFolder + 'sin_2_ts_def_50x4000_w_grad.npz'
            self._trajectoryPath2 = self._dataFolder + 'sin_mx2_ts_def_50x4000_w_grad.npz'
        else:
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sin_infer/'
            self._trajectoryPath = self._dataFolder + 'ts_0.002_10x10000_w_grad.npz'

        print('Data Folder: ', self._dataFolder)



        obs, act, next_obs, tasks = self._load_trajectories()

        self.train_windows,self.test_windows = self._pre_process(obs, act, next_obs, tasks)

        print(
            "#####################SANITY cHECK After DownSampling and Train Test Split and Normalization and batching after returning#####################",
            self.train_windows['obs'][0, 0, :5, 1],
            self.train_windows['target'][0, 0, :5, 1])
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}



    def get_statistics(self, data, dim, difference=False):
        if difference:
            data = (data[:, 1:, :dim] - data[:, :-1, :dim])
        reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
        data = reshape(data);
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std

    def _load_trajectories(self):

        if self.c.terrain == 'both240':
            data_np1 = np.load(self._trajectoryPath1)
            data_np2 = np.load(self._trajectoryPath2)
            print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', data_np1['pos'][0,:5])

            # collect obs, act, next states
            data = {'observations': [], 'actions': [], 'next_observations': []}
            # dim = np.r_[0:15, 30:41]
            print(data_np1['pos'][:, :-1, :].shape,data_np1['orn_euler'][:, :-1, :3].shape)
            data['observations'] = np.concatenate((np.concatenate((data_np1['pos'][:, :749, :3],
                                                                    np.sin(data_np1['orn_euler'])[:, :749, :],
                                                                    np.cos(data_np1['orn_euler'])[:, :749, :]),
                                                                    axis=-1), \
                                                    np.concatenate((data_np2['pos'][:, :749, :3],
                                                                    np.sin(data_np2['orn_euler'])[:, :749, :],
                                                                    np.cos(data_np2['orn_euler'])[:, :749, :]),
                                                                    axis=-1)), axis=0)
            data['actions'] = np.concatenate(
                (data_np1['jointAppliedTorques'][:, :749, :], data_np2['jointAppliedTorques'][:, :749, :]), axis=0)
            data['tasks'] = np.concatenate((data_np1['pos'][:, :749, 3], data_np2['pos'][:, :749, 3]), axis=0)
            data['next_observations'] = np.concatenate((np.concatenate((data_np1['pos'][:, 1:750, :3],
                                                                        np.sin(data_np1['orn_euler'])[:, 1:750, :],
                                                                        np.cos(data_np1['orn_euler'])[:, 1:750, :]),
                                                                        axis=-1),
                                                        np.concatenate((data_np1['pos'][:, 1:750, :3],
                                                                        np.sin(data_np1['orn_euler'])[:, 1:750, :],
                                                                        np.cos(data_np1['orn_euler'])[:, 1:750, :]),
                                                                        axis=-1)), axis=0)
            obs = data['observations']
            print('>>>>>>>>>>>>Processed Data Trajectories with shape<<<<<<<<<<<<<<<', obs.shape,
                    data_np1['jointAppliedTorques'].shape)
            act = data['actions']
            next_obs = data['next_observations']
            tasks = data["tasks"]

        else:
            data_np = np.load(self._trajectoryPath)
            print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', data_np['pos'][49,:5])

            # collect obs, act, next states
            data = {'observations': [], 'actions': [], 'next_observations': []}
            # dim = np.r_[0:15, 30:41]
            data['observations'] = np.concatenate((data_np['pos'][:, :-1, :3],
                                                    np.sin(data_np['orn_euler'])[:, :-1, :],
                                                    np.cos(data_np['orn_euler'])[:, :-1, :]), axis=-1)
            data['actions'] = data_np['jointAppliedTorques'][:, :-1, :]
            data['tasks'] = data_np['pos'][:, :-1, 3]
            data['next_observations'] = np.concatenate((data_np['pos'][:, 1:, :3],
                                                        np.sin(data_np['orn_euler'])[:, 1:, :],
                                                        np.cos(data_np['orn_euler'])[:, 1:, :]), axis=-1)
            obs = data['observations']
            print('>>>>>>>>>>>>Processed Data Trajectories with shape<<<<<<<<<<<<<<<', obs.shape,
                    data_np['jointAppliedTorques'].shape)
            plot = False
            if plot:
                for i in range(obs.shape[0]):
                    fig, axs = plt.subplots(5, 2)
                    axs[0, 0].plot(obs[i,:1500,0])
                    axs[0, 0].set_title('Joint 1')
                    axs[0, 1].plot(obs[i,:1500,1])
                    axs[0, 1].set_title('Joint 2')
                    axs[1, 0].plot(obs[i,:1500,2])
                    axs[1, 0].set_title('Joint 3')
                    axs[1, 1].plot(obs[i,:1500,3])
                    axs[1, 1].set_title('Joint 4')
                    axs[2, 0].plot(obs[i,:1500,4])
                    axs[2, 0].set_title('Joint 5')
                    axs[2, 1].plot(obs[i,:1500,5])
                    axs[2, 1].set_title('JOint 6')
                    axs[3, 0].plot(obs[i,:1500,6])
                    axs[3, 0].set_title('Joint 7')
                    axs[3, 1].plot(obs[i, :1500, 7])
                    axs[3, 1].set_title('Joint 8')
                    axs[4, 0].plot(obs[i, :1500, 8])
                    axs[4, 0].set_title('Joint 9')
                    plt.show()
            act = data['actions']
            next_obs = data['next_observations']
            tasks = data["tasks"]



        return obs, act, next_obs, tasks




if __name__ == '__main__':
    dataFolder = os.getcwd() + '/dataFolder/MobileRobot/sin2/'
    # self._trajectoryPath = self._dataFolder + 'HalfCheetahEnv_6c2_cripple.pickle'
    trajectoryPath = dataFolder + 'ts_002_50x2000.npz'
    data = np.load(trajectoryPath)
    print(data.keys())
    print(np.sin(data['orn_euler']))
    print(np.cos(data['orn_euler']))
    print(data['orn_euler'])

