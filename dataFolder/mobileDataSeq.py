import sys
import os

sys.path.append('.')
from torch.utils.data import Dataset
import numpy as np
import pickle
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import pandas as pd
from utils.dataProcess import normalize, denormalize


class metaMobileData(Dataset):
    def __init__(self, data_cfg=None):
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
        else:
            self._dataFolder = get_original_cwd() + '/dataFolder/MobileRobot/sinMix2/'
            if self.c.frequency == '500':
                self._trajectoryPath = self._dataFolder + 'ts_002_50x2000_w_grad.npz'
            if self.c.frequency == '240':
                self._trajectoryPath = self._dataFolder + 'ts_def_50x1000_w_grad.npz'

        self._save_windows = self.c.save
        self._load_windows = self.c.load
        self._imp
        self.trajPerTask = self.c.trajPerTask

        self.tar_type = self.c.tar_type

        self._split = OmegaConf.to_container(self.c.split)
        self._shuffle_split = self.c.shuffle_split

        self.meta_batch_size = self.c.meta_batch_size
        self.batch_size = self.c.batch_size  # window length/2
        self.normalizer = None
        self.filename = self.c.file_name
        self.standardize = self.c.standardize
        self.train_windows, self.test_windows = self._load_data()
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}

    

    def get_statistics(self, data, dim, difference=False):
        if difference:
            data = (data[:, 1:, :dim] - data[:, :-1, :dim])
        reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
        data = reshape(data);
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std

    def _load_data(self):
        # load the pickle file of trajectories
        if self._load_windows is not None:
            train_data_window = pickle.load(open(self._dataFolder + self.filename + '_train.pickle', 'rb'))
            test_data_window = pickle.load(open(self._dataFolder + self.filename + '_test.pickle', 'rb'))
            self.normalizer = train_data_window['normalization']
            print('>>>>>>>>>>>>Loaded Saved Windows with shape<<<<<<<<<<<<<<<', train_data_window['obs'].shape)

        else:
            data_np = np.load(self._trajectoryPath)
            print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', data_np['pos'].shape)

            # collect obs, act, next states
            data = {'observations':[], 'actions':[], 'next_observations':[]}
            # dim = np.r_[0:15, 30:41]
            data['observations'] = np.concatenate((data_np['pos'][:,:-1,:3],np.sin(data_np['orn_euler'])[:,:-1,:],np.cos(data_np['orn_euler'])[:,:-1,:]),axis=-1)
            data['actions'] = data_np['jointAppliedTorques'][:,:-1, :]
            data['grad'] = data_np['pos'][:, :-1, 3]
            data['next_observations'] = np.concatenate((data_np['pos'][:,1:,:3],np.sin(data_np['orn_euler'])[:,1:,:],np.cos(data_np['orn_euler'])[:,1:,:]),axis=-1)
            obs = data['observations']
            print('>>>>>>>>>>>>Processed Data Trajectories with shape<<<<<<<<<<<<<<<', obs.shape,data_np['jointAppliedTorques'].shape)
            act = data['actions']
            next_obs = data['next_observations']
            grad = data["grad"]

            # train test split
            train_obs, train_act, train_next_obs, train_grad, test_obs, test_act, test_next_obs, test_grad = self.train_test_split(obs, act,
                                                                                                            next_obs,grad)
            train_delta = train_next_obs - train_obs
            test_delta = test_next_obs - test_obs

            # get different statistics for state, actions, delta_state, delta_action and residuals which will be used for standardization
            mean_state_diff, std_state_diff = self.get_statistics(train_delta, train_delta.shape[-1], difference=True)
            mean_obs, std_obs = self.get_statistics(train_obs, train_obs.shape[-1])
            mean_act, std_act = self.get_statistics(train_act, train_act.shape[-1])
            self.normalizer = dict()
            self.normalizer['observations'] = [mean_obs, std_obs]
            self.normalizer['actions'] = [mean_act, std_act]
            self.normalizer['diff'] = [mean_state_diff, std_state_diff]

            # compute delta
            if self.tar_type == 'delta':
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Training On Differences <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.normalizer['targets'] = [mean_state_diff, std_state_diff]
            else:
                print(
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>> Training On Next States(not differences) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.normalizer['targets'] = [mean_obs, std_obs]

            # Standardize
            if self.standardize:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>Standardizing The Data<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

                self.train_obs = normalize(train_obs, self.normalizer["observations"][0],
                                                self.normalizer["observations"][1])
                self.train_acts = normalize(train_act, self.normalizer["actions"][0],
                                                 self.normalizer["actions"][1])

                self.test_obs = normalize(test_obs, self.normalizer["observations"][0],
                                               self.normalizer["observations"][1])
                self.test_acts = normalize(test_act, self.normalizer["actions"][0],
                                                self.normalizer["actions"][1])

                if self.tar_type == 'delta':
                    self.train_targets = normalize(train_delta, self.normalizer["diff"][0],
                                                        self.normalizer["diff"][1])
                    self.test_targets = normalize(test_delta, self.normalizer["diff"][0],
                                                       self.normalizer["diff"][1])
                else:
                    self.train_targets = normalize(train_next_obs, self.normalizer["observations"][0],
                                                        self.normalizer["observations"][1])
                    self.test_targets = normalize(test_next_obs, self.normalizer["observations"][0],
                                                       self.normalizer["observations"][1])


            else:
                self.train_obs = train_obs
                self.train_acts = train_act
                self.test_obs = test_obs
                self.test_acts = test_act
                if self.tar_type == 'delta':
                    self.train_targets = train_delta
                    self.test_targets = test_delta
                else:
                    self.train_targets = train_next_obs
                    self.test_targets = test_next_obs
            self.train_task_idx = train_grad
            self.test_task_idx = test_grad

            # Get random windows
            train_data_window = self._get_batch(train=True)
            test_data_window = self._get_batch(train=False)
            if self._save_windows is not None:
                pickle.dump(train_data_window, open(self._dataFolder + self.filename + '_train.pickle', 'wb'))
                pickle.dump(test_data_window, open(self._dataFolder + self.filename + '_test.pickle', 'wb'))

        if isinstance(self._shuffle_split, float):
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
            train_set['target'] = train_data_window['target'][idx_train,:,:3];
            test_set['target'] = train_data_window['target'][idx_test,:,:3];
            train_set['task_index'] = train_data_window['task_index'][idx_train,:,3];
            test_set['task_index'] = train_data_window['task_index'][idx_test,:,3];
            train_set['normalizer'] = self.normalizer;
            test_set['normalizer'] = self.normalizer
            print('Train Test Split Ratio', self._shuffle_split)
            return train_set, test_set

        return train_data_window, test_data_window


    def _get_batch(self, train, percentage_imputation=0.0):
        # Takes multiple paths and splits them into windows based on random locations within a trajectory
        if train:
            num_paths, len_path = self.train_obs.shape[:2]
            idx_path = np.random.randint(0, num_paths,
                                         size=self.meta_batch_size)  # task index, which gets mixed along the
            # process
            idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size)

            obs_batch = np.array([self.train_obs[ip,
                                  ib - self.batch_size:ib + self.batch_size, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            act_batch = np.array([self.train_acts[ip,
                                  ib - self.batch_size:ib + self.batch_size, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            target_batch = np.array([self.train_targets[ip,
                                     ib - self.batch_size:ib + self.batch_size, :]
                                     for ip, ib in zip(idx_path, idx_batch)])
            t_idx_batch = np.array([self.train_task_idx[ip,
                                    ib - self.batch_size:ib + self.batch_size]
                                    for ip, ib in zip(idx_path, idx_batch)])

        else:
            num_paths, len_path = self.test_obs.shape[:2]
            idx_path = np.random.randint(0, num_paths, size=self.meta_batch_size)
            idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size)

            obs_batch = np.array([self.test_obs[ip,
                                  ib - self.batch_size:ib + self.batch_size, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            act_batch = np.array([self.test_acts[ip,
                                  ib - self.batch_size:ib + self.batch_size, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            target_batch = np.array([self.test_targets[ip,
                                     ib - self.batch_size:ib + self.batch_size, :]
                                     for ip, ib in zip(idx_path, idx_batch)])
            t_idx_batch = np.array([self.test_task_idx[ip,
                                  ib - self.batch_size:ib + self.batch_size]
                                  for ip, ib in zip(idx_path, idx_batch)])

        rs = np.random.RandomState(seed=42)
        obs_valid_batch = rs.rand(obs_batch.shape[0], obs_batch.shape[1], 1) < 1 - percentage_imputation
        obs_valid_batch[:, :5] = True

        data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid': obs_valid_batch,
                        'normalization': self.normalizer,
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

        assert len(idx_train) + len(idx_test) <= episodes

        return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], grad[idx_train, :], \
               obs[idx_test, :], act[idx_test, :], delta[idx_test, :], grad[idx_test, :]


if __name__ == '__main__':
    dataFolder = os.getcwd() + '/dataFolder/MobileRobot/sin2/'
    # self._trajectoryPath = self._dataFolder + 'HalfCheetahEnv_6c2_cripple.pickle'
    trajectoryPath = dataFolder + 'ts_002_50x2000.npz'
    data = np.load(trajectoryPath)
    print(data.keys())
    print(np.sin(data['orn_euler']))
    print(np.cos(data['orn_euler']))
    print(data['orn_euler'])

