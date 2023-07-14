import numpy as np
import tqdm
from utils.dataProcess import diffToState, get_ctx_target_multistep
import torch


class longHorizon_Seq():
    def __init__(self, X_test, X_action, X_target, model, numJ, data, steps=3, horizon=100, mbrl_style='False', type='delta', standardize = True,
                 variance=True):
        '''

        :param X_test: Numpy array of inputs
        :param X_action: Numpy array of actions
        :param model:trained model object keras/Gpy/sklearn etc
        :param numJ: number of joints (input dim=3*numJ, output=2*numJ)
        :param steps: intiger for the n step prediction
        :param variance: if predictions give variance estimates(rg:GPy) in addition to mean set to True

        =====
        Outputs: Unnormalized differences if 'Delta', normalized next state if 'NextState'
        '''

        self.X_test = X_test
        self.X_act = X_action
        self.X_target = X_target
        self.model = model
        self.steps = steps - 1
        self.mbrl_style = mbrl_style
        self.horizon = horizon
        self.current_pred = None
        self.current_inp = None
        self.pred = np.zeros(
            (self.X_test.shape[0], self.X_test.shape[1] - self.horizon - self.steps + 1, self.X_test.shape[2]))
        self.numJ = numJ
        self.type = type
        self.standardize = standardize
        self.variance = variance
        self.data = data
        if self.type == 'delta':
            self.pred = np.zeros(
                (self.X_test.shape[0], self.X_test.shape[1] - self.horizon - self.steps + 1, self.X_test.shape[2]))

    def multistep(self):
        self.t = 0
        t = self.t
        ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
            get_ctx_target_impute(X_test, X_act, X_target, k, num_context=None, tar_burn_in=5,
                                  random_seed=True)
        predict(self, obs: torch.Tensor, act: torch.Tensor, obs_valid: torch.Tensor, y_context: torch.Tensor,
        batch_size: int = -1, multiStep = 0)
        if self.type == 'delta':
            # print(self.X_window.shape, self.X_action.shape, self.valid_Window.shape)

            self.diff_pred, _ = self.model.predict(self.X_window, self.X_action, self.valid_Window, self.X_window.shape[0])
            self.diff_pred = self.diff_pred[:, :,
                             :self.X_window.shape[2]]
            self.diff_pred = self.diff_pred.cpu().detach().numpy()
            self.current_pred = self._diffToStateWrapper()
        else:
            self.current_pred, _ = self.model.predict(self.X_window, self.X_action, self.valid_Window, self.X_window.shape[0])
            self.current_pred = self.current_pred[:, :, :self.X_window.shape[2]]
            self.current_pred = self.current_pred.cpu().detach().numpy()
        self.pred[:, self.t, :] = self.current_pred[:, -1, :]
        self.t = t + 1
        t = t + 1
        pbar.update(1)
        # print('between correct',self.pred.shape)
        pbar.close()
        if self.type == 'delta':
            print('predicted shape', self.pred.shape)
            self.true, _ = diffToState(self.X_target[:, self.horizon + self.steps - 1:, :],
                                       self.X_test[:, self.horizon + self.steps - 1:, :], self.data,self.standardize)
            # plt.plot(self.X_window[1, (self.horizon):, 7], label='un Win')
            # plt.plot(self.true[1,: , 7], label='n Win')

        else:
            self.true = self.X_target[:, self.horizon + self.steps - 1:, :]
            self.true = self.true.cpu().detach().numpy()

        return self.pred[:, :, :], self.true[:, :, :]

    def multistep_mbrl(self):
        self.t = 0
        t = self.t
        pbar = tqdm.tqdm(total=self.X_test.shape[1] - self.horizon - self.steps)
        while (t + self.horizon + self.steps - 1 < self.X_test.shape[1]):
            # At each time step make first self.horizon obs_valid flags True, next n-1 false
            # effectively creating windows that we slide as j increases
            t = self.t
            self.valid_Window = torch.from_numpy(np.expand_dims(
                np.array([(t + self.horizon) * [True] + (self.steps) * [False]] * self.X_test.shape[0]), 2)).bool()

            self.X_window = self.X_test[:, :(t + self.horizon + self.steps)]
            self.X_action = self.X_act[:, :(t + self.horizon + self.steps)]

            if self.type == 'delta':
                # print(self.X_window.shape, self.X_action.shape, self.valid_Window.shape)

                self.diff_pred, _ = self.model.predict(self.X_window, self.X_action, self.valid_Window, self.X_window.shape[0])
                self.diff_pred = self.diff_pred[:, :,
                                 :self.X_window.shape[2]]
                self.diff_pred = self.diff_pred.cpu().detach().numpy()
                self.current_pred = self._diffToStateWrapper()
            else:
                self.current_pred, _ = self.model.predict(self.X_window, self.X_action, self.valid_Window, self.X_window.shape[0])
                self.current_pred = self.current_pred[:, :, :self.X_window.shape[2]]
                self.current_pred = self.current_pred.cpu().detach().numpy()
            self.pred[:, self.t, :] = self.current_pred[:, -1, :]
            self.t = t + 1
            t = t + 1
            pbar.update(1)
        # print('between correct',self.pred.shape)
        pbar.close()
        if self.type == 'delta':
            print('predicted shape', self.pred.shape)
            self.true, _ = diffToState(self.X_target[:, self.horizon + self.steps - 1:, :],
                                       self.X_test[:, self.horizon + self.steps - 1:, :], self.data,self.standardize)
            # plt.plot(self.X_window[1, (self.horizon):, 7], label='un Win')
            # plt.plot(self.true[1,: , 7], label='n Win')

        else:
            self.true = self.X_target[:, self.horizon + self.steps - 1:, :]
            self.true = self.true.cpu().detach().numpy()

        return self.pred[:, :, :], self.true[:, :, :]

    def _diffToStateWrapper(self):
        pred_state = np.zeros(self.diff_pred.shape)
        pred_state[:, :(self.t + self.horizon), :self.X_window.shape[2]], _ = diffToState(
            self.diff_pred[:, :(self.t + self.horizon), :self.X_window.shape[2]],
            self.X_window[:, :(self.t + self.horizon), :self.X_window.shape[2]], self.data,self.standardize)
        # predN = denorm(pred_state,self.data,'observations')
        # # if self.t==100:
        # #     # plt.plot(self.X_window[1, :(self.t + self.horizon ), 7],label='un Win')
        # #     # plt.plot(self.X_winN[1, :(self.t + self.horizon), 7],label='n Win')
        # #     # plt.plot(pred_state[1, :(self.t + self.horizon - 1), 7],label='pre U')
        # #     # plt.plot(predN[1, :(self.t + self.horizon - 1), 7],label='pre U')
        # #     # plt.legend()
        #     # plt.show()
        j = 0
        while (j < self.steps):
            # print('hello',self.t,j)
            # if self.t==1 and j==0 :
            # plt.plot(self.X_window[1, :(self.t + self.horizon + j), 7])
            # plt.plot(pred_state[1,:(self.t+self.horizon+j-1),7])
            # plt.show()
            pred_state[:, self.t + self.horizon + j, :self.X_window.shape[2]], _ = diffToState(
                self.diff_pred[:, self.t + self.horizon + j, :self.X_window.shape[2]],
                pred_state[:, self.t + self.horizon + j - 1, :self.X_window.shape[2]], self.data,self.standardize)
            j = j + 1

        return pred_state
