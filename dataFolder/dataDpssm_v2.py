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


class metaData(Dataset):
    def __init__(self, data_cfg=None):
        if data_cfg is None:
            raise Exception('Please specify a valid Confg for data')
        else:
            self.c = data_cfg
            self.c = data_cfg

        self._long_term_pred = self.c.long_term_pred

        self._save_windows = self.c.save
        self._load_windows = self.c.load
        self.trajPerTask = self.c.trajPerTask
        self._down_sample = self.c.downsample

        self.tar_type = self.c.tar_type
        self._obs_impu = self.c.obs_imp
        self._task_impu = self.c.task_imp

        self._split = OmegaConf.to_container(self.c.split)
        self._shuffle_split = self.c.shuffle_split

        self.num_training_sequences = self.c.num_training_sequences
        self.num_testing_sequences = self.c.num_testing_sequences
        self.episode_length = self.c.episode_length  # window length for a single hiprssm instance
        self.num_episodes = self.c.num_episodes # number of hiprssm instances
        self.normalizer = None
        self.filename = self.c.file_name
        self.standardize = self.c.standardize
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
        """
                Load from disk the obs, act, next_obs, task_labels
                """
        return NotImplementedError

    def _pre_process(self, obs, act, next_obs, tasks):
        ### Downsample
        obs = obs[:,::self._down_sample,:]
        act = act[:,::self._down_sample,:]
        next_obs = next_obs[:,::self._down_sample,:]
        # train test split
        train_obs, train_act, train_next_obs, train_tasks, test_obs, test_act, test_next_obs, test_tasks = self.train_test_split(
            obs, act,
            next_obs, tasks)
        train_delta = train_next_obs - train_obs
        test_delta = test_next_obs - test_obs

        # get different statistics for state, actions, delta_state, delta_action and residuals which will be used for standardization
        mean_state_diff, std_state_diff = self.get_statistics(train_delta, dim=train_delta.shape[-1], difference=False)
        mean_obs, std_obs = self.get_statistics(train_obs, dim=train_obs.shape[-1])
        mean_act, std_act = self.get_statistics(train_act, dim=train_act.shape[-1])
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
        self.train_task_idx = train_tasks
        self.test_task_idx = test_tasks

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
            train_set['obs'] = train_data_window['obs'][idx_train, :, :3];
            test_set['obs'] = train_data_window['obs'][idx_test, :, :3];
            train_set['act'] = train_data_window['act'][idx_train];
            test_set['act'] = train_data_window['act'][idx_test];
            train_set['target'] = train_data_window['target'][idx_train, :, :3];
            test_set['target'] = train_data_window['target'][idx_test, :, :3];
            train_set['task_index'] = train_data_window['task_index'][idx_train, :, 3];
            test_set['task_index'] = train_data_window['task_index'][idx_test, :, 3];
            train_set['normalizer'] = self.normalizer;
            test_set['normalizer'] = self.normalizer
            print('Train Test Split Ratio', self._shuffle_split)
            return train_set, test_set

        return train_data_window, test_data_window

    def _get_batch(self, train):
        # Takes multiple paths and splits them into windows based on random locations within a trajectory
        if train:
            num_paths, len_path = self.train_obs.shape[:2]
            print("..............................", self.train_obs.shape)
            idx_path = np.random.randint(0, num_paths,
                                         size=self.num_training_sequences)  # task index, which gets mixed along the
            # process
            idx_batch = np.random.randint(0, len_path - self.num_episodes*self.episode_length, size=self.num_training_sequences)
            obs_batch = np.array([self.train_obs[ip,
                                  ib:ib + self.num_episodes*self.episode_length, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            #obs_batch = np.transpose(np.array(np.split(obs_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            act_batch = np.array([self.train_acts[ip,
                                  ib:ib + self.num_episodes*self.episode_length, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            #act_batch = np.transpose(np.array(np.split(act_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            target_batch = np.array([self.train_targets[ip,
                                     ib:ib + self.num_episodes*self.episode_length, :]
                                     for ip, ib in zip(idx_path, idx_batch)])
            #target_batch = np.transpose(np.array(np.split(target_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            t_idx_batch = np.array([self.train_task_idx[ip,
                                    ib:ib + self.num_episodes*self.episode_length]
                                    for ip, ib in zip(idx_path, idx_batch)])
            #t_idx_batch = np.transpose(np.array(np.split(t_idx_batch, self.num_episodes,axis=1)),axes=[1,0,2])

        else:
            num_paths, len_path = self.test_obs.shape[:2]
            idx_path = np.random.randint(0, num_paths, size=self.num_testing_sequences)
            idx_batch = np.random.randint(0, len_path - self.num_episodes*self.episode_length, size=self.num_testing_sequences)

            obs_batch = np.array([self.test_obs[ip,
                                  ib:ib + self.num_episodes*self.episode_length, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            #obs_batch = np.transpose(np.array(np.split(obs_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            act_batch = np.array([self.test_acts[ip,
                                  ib:ib + self.num_episodes*self.episode_length, :]
                                  for ip, ib in zip(idx_path, idx_batch)])
            #act_batch = np.transpose(np.array(np.split(act_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            target_batch = np.array([self.test_targets[ip,
                                     ib:ib + self.num_episodes*self.episode_length, :]
                                     for ip, ib in zip(idx_path, idx_batch)])
            #target_batch = np.transpose(np.array(np.split(target_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            t_idx_batch = np.array([self.test_task_idx[ip,
                                  ib:ib + self.num_episodes*self.episode_length]
                                  for ip, ib in zip(idx_path, idx_batch)])

            #t_idx_batch = np.transpose(np.array(np.split(t_idx_batch,self.num_episodes,axis=1)),axes=[1,0,2])

        rs = np.random.RandomState(seed=42)
        obs_valid_batch = rs.rand(obs_batch.shape[0], obs_batch.shape[1],  1) < 1 - self._obs_impu
        obs_valid_batch[:, :5] = True


        if self._long_term_pred > 1:
            #set all task_valid flags to 1.
            #task_valid_batch = rs.rand(obs_batch.shape[0], obs_batch.shape[1], 1) < 1
            #false_length = int(obs_batch.shape[1] * self._task_impu)
            #true_length = obs_batch.shape[1] - false_length
            #task_valid_batch[:, :-self._long_term_pred] = True
            #task_valid_batch[:, -self._long_term_pred:] = False
            #set obs_valid for meta-step to False when task_valid is false (to verify)
            #TODO Verify
            obs_valid_batch[:, -self._long_term_pred:, :] = False

        task_valid_batch = rs.rand(obs_batch.shape[0], self.num_episodes, 1) < 1 - self._task_impu
        task_valid_batch[:,:1] = True




        data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid': obs_valid_batch, "task_valid": task_valid_batch,
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

