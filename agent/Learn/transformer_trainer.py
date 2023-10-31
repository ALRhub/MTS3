import os
import time
from typing import Tuple
import datetime


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb

from meta_dynamic_models.neural_process_dynamics.npDynamics import npDyn
from utils.dataProcess import split_k_m, get_ctx_target_impute, nearest_ones_distance, nearest_ones_distance_3d
from utils.Losses import mse, mae, gaussian_nll , CrossEntropy, kl_rmse
from utils.PositionEmbedding import PositionEmbedding as pe
from utils import ConfigDict
from utils.plotTrajectory import plotImputation
from utils.latentVis import plot_clustering
import os
import subprocess
from hydra.utils import get_original_cwd

optim = torch.optim
nn = torch.nn

import random
import torch
import gc




def avg_pred(scores):
    '''
    scores: 2D matrices of batch_size*classes. each rows contains the score of each class
    return: convert the scores to probability using softmax and then take expected value
    '''
    scores  = scores.float()
    softmax = nn.Softmax(dim=-1)
    probs   = softmax(scores)
    classes = torch.arange(scores.size()[-1]).float()
    avg     = torch.matmul(probs,classes)
    return torch.unsqueeze(avg,dim=-1)
#avg_pred(torch.tensor([[0, 0, 0, 5, 5],[5, 5, 0, 0, 0],[1, 0, 5, 0, 0]]))


def k_max_class(scores2, k_ind=5):
    '''
    receive raw scores and take the k_ind max of them and replace the rest with  "-Inf"
    scores2: 2D matrices of batch_size*classes. each rows contains the score of each class
    k_ind: how many of the scores should be considered (rest of scores will be replaced by -inf to have prob of 0)

    '''

    vals_max, inds = torch.topk(scores2,k_ind)  # inds in each rows show the first k_ind max valuse #we dont need vals_max
    one_hot = nn.functional.one_hot(inds, num_classes=scores2.size()[-1])
    to_be_taken_inds = torch.sum(one_hot, dim=-2)  # to have all ones (to be take inds) in one vector
    cut_scores = torch.mul(scores2, to_be_taken_inds).float()  #
    cut_scores[cut_scores == 0] = float("-Inf")
    return cut_scores


# a = torch.tensor([[0, 0, 0, 5, 5], [5, 5, 0, 0, 0], [1, 0, 5, 0, 0]])
# new_score = k_max_class(a)
# print(new_score)
# print(avg_pred(new_score))


class Learn:

    def __init__(self, model, loss: str, imp: float = 0.0, config: ConfigDict = None, run = None, log=True, use_cuda_if_available: bool = True  ,normalizer=None):
        """
        :param model: nn module for np_dynamics
        :param loss: type of loss to train on 'nll' or 'mse' or 'mae' (added by saleh) or 'CrossEntropy' (added by saleh)
        :param imp: how much to impute
        :param use_cuda_if_available: if gpu training set to True
        """
        torch.cuda.empty_cache()
        assert run is not None, 'pass a valid wandb run'

        self._loss = loss

        self._model = model

        self._exp_name = run.name #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print("trainer line 51:    self._exp_name = ",self._exp_name)
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config

        #self._use_cuda = self.c.data_reader.use_cuda
        #self._device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
        self._device = torch.device("cpu")
        #self._pe = pe(self._device)

        self._imp = self.c.data_reader.imp
        self._tar_type=self.c.data_reader.tar_type   #saleh_added

        self._half_context_1     = self.c.data_reader.context_size
        self._half_context_2     = self.c.data_reader.burn_in_size # I didnt want to change the name because of the short time// otherwise the actual burn_in=0

        #print("self._tar_type = ", self._tar_type)
        self._normalizer = normalizer

        self._learning_rate = self.c.learn.lr

        self._run_name = run.name
        self._run_id   = run.id



        #local_save path:
        self._save_path_local = os.path.join( get_original_cwd() , 'experiments/saved_models' , str(self._run_name), str(self._run_id) + '.ckpt' )
        print("line 103 trainer.py:  self._save_path_local= ",self._save_path_local)


        directory = os.path.dirname(self._save_path_local)
        if not os.path.exists(directory):
            os.makedirs(directory)





        self._cluster_vis = self.c.learn.latent_vis

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self._log = bool(log)
        self.save_model = self.c.learn.save_model
        if self._log:
            self._run = run

    def train_step(self, train_obs: np.ndarray, train_act: np.ndarray, train_targets: np.ndarray,batch_size: int , cfg=None , train_task_idx: np.ndarray =None)  -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions
        :param train_targets: training targets
        :param train_task_idx: task ids per episode
        :param batch_size: batch size for each gradient update
        :return: average loss (nll) and  average metric (rmse), execution time
        """
        time_list=[]
        start = time.time()
        self._train_with_impute = cfg.data_reader.train_with_impute
        #print("train_with_impute:",self._train_with_impute)
        avg_loss = avg_metric_nll = avg_metric_mse= avg_metric_packet_mse = avg_metric_mae= avg_metric_CrossEntropy = avg_metric_combined_kl_rmse= avg_metric_kl  = 0


        k = cfg.data_reader.context_size
        k2 = cfg.data_reader.burn_in_size  # not used as a burn_in. it is just used to have 75+75=150 obs_valid=True
        burn_in = cfg.data_reader.burn_in_size
        m = cfg.data_reader.target_size

        self._model.train()
        dataset = TensorDataset(train_obs, train_act, train_targets)
        #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2  , collate_fn = my_collate)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


        t0 = time.time()
        #b = list(loader)[0]
        z_vis_list = []
        task_id_list = []

        print("line 146 transformer_trainer.py : Device = ", self._device)

        for batch_idx, (obs, act, targets) in enumerate(loader):
            # Assign tensors to device
            #print("batch_idx:",batch_idx)
            obs_batch = (obs)#.to(self._device)

            act_batch = act#.to(self._device)
            #print("batch_idx:",batch_idx)

            target_batch = (targets)#.to(self._device)
            #task_id = (task_id).to(self._device)

            ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, tar_imp=self._imp,random_seed=True)

            ctx_inp = torch.cat([ctx_obs_batch,    ctx_act_batch] , dim=-1).to(self._device)   #obs: -----ctx_obs-------/-------tar_obs------
            tar_inp = torch.cat([tar_obs_batch,    tar_act_batch] , dim=-1).to(self._device)   #tar:  -----ctx_tar-------/-------tar_tar------

            X = torch.cat([ctx_inp,    tar_inp] , dim=1).to(self._device)
            #Y = tar_tar_batch[:,burn_in:,:].to(self._device) # if we want to predict only the tar_tar, howerver vaishak asked to also predict the context
            #Y = torch.cat([ ctx_target_batch, tar_tar_batch] , dim=1) # we want to predict ctx_tar as well as tar_tar ---> vaishak said!
            Y = torch.cat([ ctx_obs_batch, tar_obs_batch] , dim=1)[:,1:,:] .to(self._device)
            # plot kon bebin chejourie ---> noise had been added, therefore cant be correctly compared


            # print("X.shape:",X.shape)   #X.shape:torch.Size([250, 900, 13])
            # print("Y.shape:",Y.shape)  #gt.shape: torch.Size([250, 750, 9])
                                        # pred_logits.shape torch.Size([250, 750, 9])

            ctx_obs_valid_batch = torch.ones(X.shape[0],self._half_context_1+self._half_context_2, 1)
            ctx_obs_valid_batch = ctx_obs_valid_batch.bool() ## .to(self._device)

            tar_obs_valid_batch   = torch.from_numpy(tar_obs_valid_batch).bool() ## .to(self._device)
            final_obs_valid_batch = torch.cat([ctx_obs_valid_batch,tar_obs_valid_batch],dim=1)#.to(self._device)

            if batch_idx == 0:
                try:
                    nonzeros = torch.count_nonzero(final_obs_valid_batch) / torch.prod(torch.tensor(final_obs_valid_batch.size()))   #works with latest version of pytorch
                except:
                    nonzeros = np.count_nonzero(final_obs_valid_batch) / np.prod(final_obs_valid_batch.shape)
                print("Fraction of Valid Target Observations:", nonzeros)


            # Set Optimizer to Zero
            self._optimizer.zero_grad()

            ###################################################  Start of the Edit ###################################################
            if(self._train_with_impute):
                # print("train with impute ...")
                # print("final_obs_valid_batch.shape",final_obs_valid_batch.shape)
                assert (X.shape[1] == final_obs_valid_batch.shape[1])
                distances = nearest_ones_distance_3d(final_obs_valid_batch).cpu()
                #print("distances:",distances)
                max_dist = torch.max(distances).cpu()
                # print('max_dist:', max_dist, "you need to run the forward pass", max_dist + 1, " times")
                torch.cuda.empty_cache()

                #Hi
                obs_dim=Y.shape[-1] #9 , X.shape[-1]=13
                act_dim=X.shape[-1] - Y.shape[-1] #13-9=4
                all_acions=X[:,:,obs_dim:]
                out_feature = X.shape[-1]  #should be obs_dim + act_dim to be able to store action and do the next inference
                # Apply f(x) based on distances
                # print("X.shape:  ", X.shape)
                B1, T1, in_feat = X.shape
                result_list_3d = []  # includes 3d tensors of [T0(x),T1(x),...Tm(x)]
                #result_arr = torch.zeros((X.shape[0], max_dist + 1, X.shape[1], out_feature)).to(self._device)
                # print("result_arr.shape:", result_arr.shape)
                #print("X:", X)
                # print("check all device...")
                # print("X.device:",X.device)
                # print("Y.device:", Y.device)
                # print("final_obs_valid_batch .device:", final_obs_valid_batch .device)
                # print("ALL_ACTIONS.device:", all_acions.device)


                #Calcualte the forward-pass max_dist+1  times
                for i in range(max_dist + 1):
                    # print("----------------------------------------------------------------------------------------------------")
                    print("iteration:", i)
                    tmp_actions = torch.zeros_like(all_acions)
                    # print("tmp_actions.device:", tmp_actions.device)
                    if (i == 0):
                        y_obs_init,_ = self._model(X)
                        # print("X.dtype",X.dtype)
                        # print("y_init",y_obs_init.dtype)
                        # print(y_init)
                        # print("f(X).shape:", y_obs_init.shape)
                        # print("result_arr[:,i,:,:].shape:", result_arr[:, i, :, :].shape)
                        #result_arr[:, i, :, :obs_dim] = y_obs_init
                        # print("y_obs_init.device",y_obs_init.device)
                        tmp_actions[:,:-1,:]=all_acions[:,1:,:] #B,T,4 // the rest are zero and they are not important
                        init_obs_act  =torch.cat([y_obs_init,tmp_actions] ,dim=-1)

                        # print("init_obs_act.shape",init_obs_act.shape)
                        # print("init_obs_act.device",init_obs_act.device)

                        #result_arr[:, i, :, obs_dim:] = tmp_actions.to(self._device)
                        result_list_3d.append(init_obs_act)

                        init_obs_act = None
                        y_obs_init   = None
                        tmp_actions  = None
                        del init_obs_act, y_obs_init , tmp_actions
                        gc.collect()
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        #print("result_arr[:, i - 1, :, :].dtype", result_arr[:, i - 1, :, :].dtype)

                        y_obs_i, _ = self._model(result_list_3d[-1])

                        # print("y_i",y_i.dtype)
                        # print("f(X).shape:", y_i.shape)
                        # print("result_arr[:,i,:,:].shape:", result_arr[:,i,:,:].shape)
                        #result_arr[:, i, :, :obs_dim] = y_i
                        tmp_actions[:, :-(i+1), :] = all_acions[:, (i+1):, :]  # rest is set to zero and they are not important
                        obs_act_i = torch.cat( [y_obs_i,tmp_actions] , dim=-1)
                        result_list_3d.append(obs_act_i)

                        y_obs_i = None
                        tmp_actions = None
                        obs_act_i = None
                        del y_obs_i, tmp_actions , obs_act_i
                        gc.collect()
                        torch.cuda.empty_cache()
                        gc.collect()

                        # result_arr[:, i, :, obs_dim:] = tmp_actions.to(self._device)
                    # print(result_arr)

                print("------------------------------************Finished Necessary Calculation***************-------------------------------")
                print("------------------------------************Start assigning***************----------------------------------------------")

                        #print('ans_intermed.shape:', result_arr.shape)

                #Hi
                # Create y by selecting elements based on mask
                B2, T2, _ = distances.shape
                assert (B1==B2 and T1==T2)
                pred_with_impute = torch.zeros((B2, T2, out_feature)).to(self._device)
                #y_hat = np.zeros((B2, T2, out_feature))

                for b in range(B2):
                    for t in range(T2):
                        u = distances[b, t, 0]  # e.g 0,1,2,3,.. max e.g 4   which shows we should use T0 or T1 or T2 or ... Tm
                        #pred_with_impute[b, t, :] = result_arr[b, u, t - u, :]
                        pred_with_impute[b, t, :] = result_list_3d[u][b, t - u, :]

                # obs_act_i = None
                # y_obs_i = None
                # init_obs_act = None
                # tmp_actions=None
                del result_list_3d
                gc.collect()
                torch.cuda.empty_cache()
                                                                                # why t-u? because T0(x) = [y_hat_0, y_hat_1, y_hat_2, y_hat_3, ... , y_hat_899]
                                                                                # why t-u? because T1(x) = [y_hat_1, y_hat_2, y_hat_3, y_hat_4, ... , y_hat_900]
                                                                                # why t-u? because T2(x) = [y_hat_2, y_hat_3, y_hat_4, y_hat_5, ... , y_hat_901]
                                                                                # why t-u? because T3(x) = [y_hat_3, y_hat_4, y_hat_5, y_hat_6, ... , y_hat_902]
                                                                                #  so it is because of the shift

                #print("look at the final pred  and its shape here")
                pred_logits = pred_with_impute[:,:-1,:obs_dim] # drop action and also the last time-step because we dont have the gt for that

            else: # normal stuff as I did previously
                #print('train without imputation')
                pred_logits, _ = self._model(X)
                pred_logits = pred_logits[:, :-1,:]  # IT DOES 1 STEP PREDICTION: -1 because we dont have the gt for the last one (decided not to use target!)

            #####################################    End of the edit          #################################



            # Forward Pass

            # pred_logits,_ = self._model(X)
            # pred_logits = pred_logits[:,:-1,:] # IT DOES 1 STEP PREDICTION: -1 because we dont have the gt for the last one (decided not to use target!)
            #pred_logits = pred_logits[:,k+burn_in:,:] # vaishak said we also need to do ctx prediction as well
            #print("pred_logits.shape",pred_logits.shape)




            if self._loss == 'nll':
                # update
                B,T,C = obs_batch.shape
                assert pred_logits.shape[-1]==2*obs_batch.shape[-1] , "mean and var shaould have the shape of " + str(obs_batch.shape[-1])+"but pred + var have the shape of  " + str(pred_logits.shape[-1])
                out_mean = pred_logits[:,:,:C]
                out_var  = pred_logits[:,:,C:]
                loss = gaussian_nll(Y, out_mean, out_var)
            elif self._loss == 'mae':
                loss = mae(pred_logits, Y)
            elif self._loss == 'mse':
                #loss = mse(tar_tar_batch, out_mean)
                #mseloss = nn.MSELoss()
                #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>LOSS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                #print("pred_logits.shape" ,pred_logits.shape)
                #print("Y.shape",Y.shape)
                loss = mse(pred_logits, Y)
                #print('loss:',loss)
                #print("loss.item():",loss.item())
            elif self._loss == 'cross_entropy':
                #print("!!!PLEASE DOUBLE CHECK INPUT AND TARGET") #done
                loss = CrossEntropy(pred_logits , Y , C=20, manual=False)
            elif self._loss == 'modified_mse':
                pass
            elif self._loss == 'kl_rmse':  # lambda1*KL(P||Q) + lambda2*mse
                loss,distr_kl,_ = kl_rmse(pred_logits,Y,lambda1=0.5,lambda2=0.5,normalization= self._normalizer, should_normalize = not(self.c.data_reader.standardize))
            else:
                raise NotImplementedError


            # Backward Pass
            loss.backward()
            #print("hiprssm_dyn_trainer.py line 146 loss.backward()")

            # Clip Gradients
            if self.c.np.clip_gradients:
                torch.nn.utils.clip_grad_norm(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()

            if (self._loss  in ['cross_entropy']):
                scores = k_max_class(out_mean) #newly added
                k_averaged = avg_pred(scores) #newly added
                out_mean_pred = out_mean


            if (self._loss == 'cross_entropy'):
                out_mean_pred = torch.argmax(out_mean, dim=-1).unsqueeze(dim=-1)  #yek adad beyne 0 ta 19 bar migardune baraye dim e -1

            with torch.no_grad():  #
                try:
                    metric_nll = gaussian_nll(Y, k_averaged, out_var)
                except:
                    metric_nll = np.nan

                #update
                if(self.c.data_reader.standardize):  # we always consider the normalized error
                    if(self.c.learn.loss !='nll'):
                        metric_mse = mse( pred_logits,Y)
                        metric_packet_mse = mse( pred_logits[:, :, -1:], Y[:, :, -1:])
                        metric_mae = mae(pred_logits, Y)
                    else:
                        B, T, C = obs_batch.shape
                        assert pred_logits.shape[-1] == 2 * obs_batch.shape[-1], "mean and var shaould have the shape of " + str(obs_batch.shape[-1]) + "but pred + var have the shape of  " + str(pred_logits.shape[-1])
                        out_mean = pred_logits[:, :, :C]
                        out_var = pred_logits[:, :, C:]

                        metric_mse = mse(out_mean, Y)
                        metric_packet_mse = mse(out_mean[:, :, -1:], Y[:, :, -1:])
                        metric_mae = mae(out_mean, Y)



                else:
                    if (self.c.learn.loss != 'nll'):
                        metric_mse = mse(pred_logits,Y)/(self._normalizer['observations'][1])**2
                        metric_packet_mse = mse( pred_logits[:, :, -1:], Y[:, :, -1:])/torch.from_numpy(self._normalizer['packets'][1]).to(self._device)**2
                        metric_mae = mae( pred_logits,Y)/(self._normalizer['packets'][1])
                    else:
                        B, T, C = obs_batch.shape
                        assert pred_logits.shape[-1] == 2 * obs_batch.shape[-1], "mean and var shaould have the shape of " + str(obs_batch.shape[-1]) + "but pred + var have the shape of  " + str(pred_logits.shape[-1])
                        out_mean = pred_logits[:, :, :C]
                        out_var = pred_logits[:, :, C:]

                        metric_mse = mse(out_mean,Y)/(self._normalizer['observations'][1])**2
                        metric_packet_mse = mse( out_mean[:, :, -1:], Y[:, :, -1:])/torch.from_numpy(self._normalizer['packets'][1]).to(self._device)**2
                        metric_mae = mae( out_mean,Y)/(self._normalizer['packets'][1])




                try:
                    metric_CrossEntropy = CrossEntropy( pred_logits,Y,C=20,manual=False)
                except:
                    metric_CrossEntropy = np.nan

                try:
                    metric_combined_kl_rmse, metric_kl,_= kl_rmse( pred_logits,Y,should_normalize=not(self.c.data_reader.standardize))
                except:
                    metric_combined_kl_rmse , metric_kl= np.nan , np.nan



                #z_vis_list.append(mu_z.detach().cpu().numpy())
                #task_id_list.append(task_id.detach().cpu().numpy())
            avg_loss += loss.detach().cpu().numpy()
            try:
                avg_metric_nll += metric_nll.detach().cpu().numpy()
            except:
                avg_metric_nll = np.nan

            avg_metric_mse += metric_mse.detach().cpu().numpy()
            avg_metric_packet_mse += metric_packet_mse.detach().cpu().numpy()
            avg_metric_mae += metric_mae.detach().cpu().numpy()

            try:
                avg_metric_CrossEntropy += metric_CrossEntropy.detach().cpu().numpy()
            except:
                avg_metric_CrossEntropy = np.nan

            try:
                avg_metric_combined_kl_rmse += metric_combined_kl_rmse.detach().cpu().numpy()
            except:
                avg_metric_combined_kl_rmse = np.nan

            try:
                avg_metric_kl += metric_kl.detach().cpu().numpy()
            except:
                avg_metric_kl = np.nan



        #print("loss")
        # taking sqrt of final avg_mse gives us rmse across an epoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))


        elif self._loss == 'mse':
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        elif self._loss == 'mae':
            avg_loss = avg_loss / len(list(loader))

        elif self._loss == 'cross_entropy':
            avg_loss = (avg_loss / len(list(loader)))

        elif self._loss == 'kl_rmse':
            avg_loss = (avg_loss / len(list(loader)))

        else:
            raise NotImplementedError



        # with torch.no_grad():
        #     self._tr_sample_gt = Y.detach().cpu().numpy() [:,:,:]  #   #packets for the sake of plots
        #     self._tr_sample_valid = tar_obs_valid_batch.detach().cpu().numpy()[:,burn_in:,:]
        #     #self._tr_sample_pred_mu = k_averaged.detach().cpu().numpy()
        #     self._tr_sample_pred_mu = pred_logits.detach().cpu().numpy()[:,:,-1:] #   #packets for the sake of plots
        #
        #     try:
        #         self._tr_sample_pred_var = out_var.detach().cpu().numpy()
        #     except:
        #         self._tr_sample_pred_var = np.nan
        #
        #     self._tr_sample_tar_obs = tar_obs_batch.detach().cpu().numpy()[:,:,-1:]   # saleh_added

        avg_metric_nll = avg_metric_nll / len(list(loader))
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))
        avg_metric_packet_rmse = np.sqrt(avg_metric_packet_mse / len(list(loader)))
        avg_metric_mae = avg_metric_mae / len(list(loader))
        avg_metric_CrossEntropy = avg_metric_CrossEntropy / len(list(loader))
        avg_metric_combined_kl_rmse     = avg_metric_combined_kl_rmse / len(list(loader))
        avg_metric_kl = avg_metric_kl / len(list(loader))

        #z_vis = 0
        try:
            z_vis = np.concatenate(z_vis_list, axis=0)
        except:
            z_vis = np.nan
        #task_labels = np.concatenate(task_id_list, axis=0)
        end = time.time()
        elapsed_time = end - start
        time_list.append(elapsed_time)
        #print("time_list:",time_list)
        return avg_loss, avg_metric_nll, avg_metric_rmse, avg_metric_packet_rmse,avg_metric_mae,avg_metric_CrossEntropy,avg_metric_combined_kl_rmse,avg_metric_kl , z_vis, None, time.time() - t0

    def eval(self, obs: np.ndarray, act: np.ndarray, targets: np.ndarray,
             batch_size: int = -1 ,cfg=None , task_idx: np.ndarray =None) -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param targets: targets to evaluate on
        :param task_idx: task index
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        #print("saleh_added: hiprssm_dyn_trainer line 199  act:",act)   #all 0 checked #saleh_added
        #print("type(act)=", type(act)) #torch


        k = cfg.data_reader.context_size
        burn_in = cfg.data_reader.burn_in_size
        m = cfg.data_reader.target_size

        #print("type(act)=",type(act))
        dataset = TensorDataset(obs, act, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        avg_loss = avg_metric_nll = avg_metric_mse= avg_metric_packet_mse = avg_metric_mae= avg_metric_CrossEntropy = avg_metric_combined_kl_rmse = avg_metric_kl = 0.0
        avg_metric = 0.0
        z_vis_list = []
        #task_id_list = []

        for batch_idx, (obs_batch, act_batch, targets_batch) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices

                #print("saleh_added: hiprssm_dyn_trainer line 196  act_batch:", act_batch)  # saleh_added #checked all_zero


                obs_batch = (obs_batch).to(self._device)
                act_batch = act_batch.to(self._device)
                target_batch = (targets_batch).to(self._device)

                # Split to context and targets
                # k = int(obs_batch.shape[1] / 2)
                # m = obs_batch.shape[1] - k

                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                    get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, tar_imp=self._imp,
                                          random_seed=True)
                #print("saleh_added hiprssm_dyn_trainer.py line 210      ctx_act_batch:",ctx_act_batch)#saleh_added #checked all 0

                #print("saleh_added hiprssm_dyn_trainer.py line 210      tar_act_batch:",tar_act_batch) #saleh_added #checked all 0


                ctx_obs_valid_batch = torch.ones(ctx_obs_batch.shape[0],ctx_obs_batch.shape[1],1)
                ctx_obs_valid_batch = ctx_obs_valid_batch.bool().to(self._device)

                tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)



                # Make context and target
                #target_X = (tar_obs_batch, tar_act_batch, tar_obs_valid_batch)
                ctx_inp = torch.cat([ctx_obs_batch, ctx_act_batch], dim=-1).to(self._device)  # obs: -----ctx_obs-------/-------tar_obs------
                tar_inp = torch.cat([tar_obs_batch, tar_act_batch], dim=-1).to(self._device)  # tar:  -----ctx_tar-------/-------tar_tar------

                X = torch.cat([ctx_inp, tar_inp], dim=1).to(self._device)
                #Y = torch.cat([ctx_target_batch, tar_tar_batch] , dim=1).to(self._device) # we want to predict ctx_tar as well as tar_tar ---> vaishak said!
                Y = torch.cat([ctx_obs_batch, tar_obs_batch], dim=1)[:, 1:, :].to(self._device) #obs shifted by 1

                # print("X.shape:", X.shape)  # X.shape: torch.Size([350, 150, 25])
                # print("gt.shape:", Y.shape)  # Y.shape: torch.Size([350, 75, 25])

                # Forward Pass
                pred_logits, _ = self._model(X)
                #pred_logits = pred_logits[:, k + burn_in:, :]  # interested in the samples after ctx+burn+in
                pred_logits = pred_logits[:, :-1, :] # it also predict the context as per vaishak's suggestion # -1 because gt for the last pred is not available
                #print("pred_logits.shape", pred_logits.shape)

                #print("pred_logits.shape:", pred_logits.shape)  # X.shape: torch.Size([350, 150, 25])



                if (self._loss in [ 'cross_entropy']):
                    scores = k_max_class(out_mean)  # newly added
                    k_averaged = avg_pred(scores)  # newly added
                    out_mean_pred = out_mean


                # if (self._loss == 'cross_entropy'):
                #     out_mean_pred = torch.argmax(out_mean, dim=-1).unsqueeze(dim=-1)  # yek adad beyne 0 ta 19 bar migardune
                #
                # self._te_sample_gt       = Y.detach().cpu().numpy() [:,:,-1:] #   #packets for the sake of plots
                # self._te_sample_valid    = tar_obs_valid_batch.detach().cpu().numpy()[:,burn_in:,:]
                #
                # self._te_sample_pred_mu  = pred_logits.detach().cpu().numpy()[:,:,-1:] #   #packets for the sake of plots
                # try:
                #     self._te_sample_pred_var = out_var.detach().cpu().numpy()
                # except:
                #     self._te_sample_pred_var = np.nan
                #
                # self._te_sample_tar_obs  = tar_obs_batch.detach().cpu().numpy()[:,:,-1:]  #added_by saleh


                ## Calculate Loss
                #update
                if self._loss == 'nll':
                    #update
                    B, T, C = obs_batch.shape
                    assert pred_logits.shape[-1] == 2 * obs_batch.shape[-1], "mean and var shaould have the shape of " + str(obs_batch.shape[-1]) + "but pred + var have the shape of  " + str(pred_logits.shape[-1])
                    out_mean = pred_logits[:, :, :C]
                    out_var = pred_logits[:, :, C:]
                    loss = gaussian_nll(Y, out_mean, out_var)


                elif self._loss == 'mse':
                    loss = mse( pred_logits,Y)

                elif self._loss == 'mae':
                    loss = mae(out_mean_pred,Y)

                elif self._loss == 'cross_entropy':   # target(GT) should come 2nd
                    loss = CrossEntropy(pred_logits,Y  ,C=20 , manual=False)

                elif self._loss == 'kl_rmse':  # lambda1*KL(P||Q) + lambda2*mse     # use it with unormalized data
                    loss,distr_kl,_ = kl_rmse(pred_logits, Y, lambda1=0.5, lambda2=0.5, normalization=self._normalizer)

                else:
                    raise NotImplementedError

                #metric_nll = gaussian_nll(tar_tar_batch, out_mean_pred, out_var)
                try:
                    metric_nll = gaussian_nll(Y, k_averaged, out_var)
                except:
                    metric_nll = np.nan

                # update
                if(self.c.data_reader.standardize):  # we always consider the normalized the error
                    if(self.c.learn.loss != 'nll'):
                        metric_mse = mse(Y, pred_logits)
                        metric_packet_mse = mse(Y[:, :, -1:], pred_logits[:, :, -1:])
                        metric_mae = mae(Y, pred_logits)
                    else:
                        B, T, C = obs_batch.shape
                        assert pred_logits.shape[-1] == 2 * obs_batch.shape[
                            -1], "mean and var shaould have the shape of " + str(
                            obs_batch.shape[-1]) + "but pred + var have the shape of  " + str(pred_logits.shape[-1])
                        out_mean = pred_logits[:, :, :C]
                        out_var = pred_logits[:, :, C:]

                        metric_mse = mse(out_mean, Y)
                        metric_packet_mse = mse(out_mean[:, :, -1:], Y[:, :, -1:])
                        metric_mae = mae(out_mean, Y)
                else:
                    if (self.c.learn.loss != 'nll'):
                        metric_mse = mse(Y, pred_logits)/(self._normalizer['observations'][1])**2
                        metric_packet_mse = mse(Y[:, :, -1:], pred_logits[:, :, -1:])/(self._normalizer['packets'][1])**2
                        metric_mae = mae(Y, pred_logits)/(self._normalizer['packets'][1])
                    else:
                        B, T, C = obs_batch.shape
                        assert pred_logits.shape[-1] == 2 * obs_batch.shape[-1], "mean and var shaould have the shape of " + str(obs_batch.shape[-1]) + "but pred + var have the shape of  " + str(pred_logits.shape[-1])
                        out_mean = pred_logits[:, :, :C]
                        out_var = pred_logits[:, :, C:]

                        metric_mse = mse(out_mean, Y)
                        metric_packet_mse = mse(out_mean[:, :, -1:], Y[:, :, -1:])
                        metric_mae = mae(out_mean, Y)




                try:
                    metric_CrossEntropy = CrossEntropy(out_mean,Y,C=20,manual=False)
                except:
                    metric_CrossEntropy = np.nan

                try:
                    metric_comined , metric_kl,  _ = kl_rmse(pred_logits, Y, lambda1=0.5, lambda2=0.5, normalization=self._normalizer)
                except:
                    metric_comined , metric_distr_kl = np.nan , np. nan




                try:
                    z_vis_list.append(mu_z.detach().cpu().numpy())
                except:
                    z_vis_list = []
                #task_id_list.append(task_id.detach().cpu().numpy())




                avg_loss += loss.detach().cpu().numpy()
                try:
                    avg_metric_nll += metric_nll.detach().cpu().numpy()
                except:
                    avg_metric_nll = np.nan


                avg_metric_mse += metric_mse.detach().cpu().numpy()
                avg_metric_packet_mse += metric_packet_mse.detach().cpu().numpy()

                avg_metric_mae += metric_mae.detach().cpu().numpy()
                try:
                    avg_metric_CrossEntropy += metric_CrossEntropy.detach().cpu().numpy()
                except:
                    avg_metric_CrossEntropy = np.nan

                try:
                    avg_metric_combined_kl_rmse += metric_comined.detach().cpu().numpy()
                except:
                    avg_metric_combined_kl_rmse= np.nan

                try:
                    avg_metric_kl += metric_kl.detach().cpu().numpy()
                except:
                    avg_metric_kl = np.nan


        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        print("line 337 dyn_trainer.py   self._loss=",self._loss)
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))

        elif self._loss == 'mse':
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        elif self._loss == 'mae':
            avg_loss = avg_loss / len(list(loader))

        elif self._loss == 'cross_entropy':
            avg_loss = avg_loss / len(list(loader))

        elif self._loss == 'kl_rmse':
            avg_loss = avg_loss / len(list(loader))

        else:
            raise NotImplementedError

        avg_metric_nll = avg_metric_nll / len(list(loader))
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))
        avg_metric_packet_rmse = np.sqrt(avg_metric_packet_mse / len(list(loader)))
        avg_metric_mae =(avg_metric_mae / len(list(loader)))
        avg_metric_CrossEntropy = (avg_metric_CrossEntropy / len(list(loader)))
        avg_metric_combined_kl_rmse = (avg_metric_combined_kl_rmse / len(list(loader)))
        avg_metric_kl = (avg_metric_kl / len(list(loader)))

        try:
            z_vis = np.concatenate(z_vis_list, axis=0)
        except:
            z_vis = 0
        #z_vis = 0
        #task_labels = np.concatenate(task_id_list, axis=0)
        return avg_loss, avg_metric_nll, avg_metric_rmse, avg_metric_packet_rmse, avg_metric_mae, avg_metric_CrossEntropy,avg_metric_combined_kl_rmse,avg_metric_kl ,z_vis, None

    def train(self, train_obs: torch.Tensor, train_act: torch.Tensor,
              train_targets: torch.Tensor, epochs: int, batch_size: int,
              val_obs: torch.Tensor = None, val_act: torch.Tensor = None,
              val_targets: torch.Tensor = None, val_task_idx: torch.Tensor = None, val_interval: int = 1,
              val_batch_size: int = -1 ,cfg=None , train_task_idx: torch.Tensor=None) -> None:
        '''
        :param train_obs: training observations for the model (includes context and targets)
        :param train_act: training actions for the model (includes context and targets)
        :param train_targets: training targets for the model (includes context and targets)
        :param train_task_idx: task_index for different training sequence
        :param epochs: number of epochs to train on
        :param batch_size: batch_size for gradient descent
        :param val_obs: validation observations for the model (includes context and targets)
        :param val_act: validation actions for the model (includes context and targets)
        :param val_targets: validation targets for the model (includes context and targets)
        :param val_task_idx: task_index for different testing sequence
        :param val_interval: how often to perform validation
        :param val_batch_size: batch_size while performing inference
        :return:
        '''

        """ Train Loop"""
        torch.cuda.empty_cache() #### Empty Cache
        if val_batch_size == -1:
            val_batch_size = 4 * batch_size
        best_loss = np.inf
        best_nll = np.inf
        best_rmse = np.inf
        best_packet_rmse = np.inf
        best_mae = np.inf
        best_CrossEntropy = np.inf
        best_combined_kl_rmse = np.inf
        best_kl = np.inf
        #print(cfg)
        #print("train_act:", train_act) #all 0 checked # saleh_added
        #print("saleh_added hiprssm_dyn_trainer.py line 318")  # saleh_added

        #print("val_act:", val_act) #all 0 checked  # saleh_added
        #print("saleh_added hiprssm_dyn_trainer.py line 322")  # saleh_added


        if self._log:
            #relogin to wandb in case in a shared cluster sb changed the wandb account
            my_key = "3f542cd12f4a9ef15e8a6e681c56c9265cbee079"
            command = f'wandb login --relogin {my_key}'
            subprocess.run(command, shell=True, check=True)


            wandb.watch(self._model, log='all')
            print("creating artifact object before training starts")

            artifact_name = str(self._run_name) +"_"+ str(self._run_id)
            print("line 637 trainer.py   artifact_name" , artifact_name)
            artifact = wandb.Artifact(name=artifact_name, type='model') #this name argument is important here and later is used for downloading
            #name (required): A string specifying the name of the artifact. This name should be unique within the scope of your project.
            #type (optional): A string specifying the type of the artifact. This can be any string you choose, and is intended to help organize artifacts by category or purpose.


        print("===========================...epoching...=================================")
        for i in range(epochs):
            print("===================================================================================")
            print("epochs = :",i , "/",epochs)



            train_loss, train_metric_nll, train_metric_rmse, train_metric_packet_rmse,train_metric_mae, train_metric_CrossEntropy,train_metric_combined_kl_rmse,train_metric_kl ,z_vis, z_labels, time_elapased = self.train_step(train_obs,
                                                                                                     train_act,
                                                                                                     train_targets,
                                                                                                     batch_size, cfg=cfg)
            print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, {}:{:.5f},{}:{:.5f}, {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, Took {:4f} seconds".format(i + 1, self._loss, train_loss, 'target_nll:', train_metric_nll, 'target_rmse:', train_metric_rmse, 'target_mae:', train_metric_mae, 'target_CrossEntropy:', train_metric_CrossEntropy, 'target_kl_rmse:', train_metric_combined_kl_rmse, 'target_kl:', train_metric_kl, 'target_packet_rmse:', train_metric_packet_rmse ,time_elapased))
            # self._writer.add_scalar(self._loss + "/train_loss", train_loss, i)
            # self._writer.add_scalar("nll/train_metric", train_metric_nll, i)
            # self._writer.add_scalar("rmse/train_metric", train_metric_rmse, i)
            if self._log:
                # my_key = "3f542cd12f4a9ef15e8a6e681c56c9265cbee079"
                # command = f'wandb login --relogin {my_key}'
                # subprocess.run(command, shell=True, check=True)
                wandb.log({self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll, "rmse/train_metric": train_metric_rmse , "packet_rmse/train_metric": train_metric_packet_rmse, "mae/train_metric": train_metric_mae, "CrossEntropy/train_metric": train_metric_CrossEntropy, "combined_kl_rmse/train_metric": train_metric_combined_kl_rmse, "kl/train_metric": train_metric_kl ,"epochs": i})


            # print("val_act:", val_act)   #saleh_added
            # print("val_targets:", val_targets)  # saleh_added
            # print("saleh_added hiprssm_dyn_trainer.py line 316")#saleh_added
            if val_obs is not None and val_targets is not None and i % val_interval == 0:
                val_loss, val_metric_nll, val_metric_rmse ,val_metric_packet_rmse ,val_metric_mae, val_metric_CrossEntropy, val_metric_combined_kl_rmse, val_metric_kl ,z_vis_val, z_labels_val = self.eval(val_obs, val_act,val_targets, batch_size=val_batch_size , cfg=cfg)
                if val_loss < best_loss:
                    if self.save_model:
                        print('>>>>>>>Saving Best Model<<<<<<<<<<')
                        torch.save(self._model.state_dict(), self._save_path_local)
                    if self._log:
                        wandb.run.summary['best_loss'] = val_loss
                    best_loss = val_loss
                if val_metric_nll < best_nll:
                    if self._log:
                        wandb.run.summary['best_nll'] = val_metric_nll
                    best_nll = val_metric_nll
                if val_metric_rmse < best_rmse:
                    if self._log:
                        wandb.run.summary['best_rmse'] = val_metric_rmse
                    best_rmse = val_metric_rmse

                if val_metric_packet_rmse < best_packet_rmse:
                    if self._log:
                        if (self._loss == 'mse' or self._loss == 'kl_rmse'):  # it means inpu and output are normalized
                            wandb.run.summary['best_packet_rmse_normalized'] = val_metric_packet_rmse
#                            wandb.run.summary['best_packet_rmse_Denormalized'] = val_metric_packet_rmse * self._normalizer['packets'][1]  #rmse_norm * std
                        elif(self._loss == 'cross_rntropy' ):
                            wandb.run.summary['best_packet_rmse_Denormalized'] = val_metric_packet_rmse
                            wandb.run.summary['best_packet_rmse_normalized'] = val_metric_packet_rmse / self._normalizer['packets'][1]  #rmse_Denorm / std



                    best_packet_rmse = val_metric_packet_rmse

                if val_metric_mae < best_mae:
                    if self._log:
                        wandb.run.summary['best_mae'] = val_metric_mae
                    best_mae = val_metric_mae

                if val_metric_CrossEntropy < best_CrossEntropy:
                    if self._log:
                        wandb.run.summary['best_CrossEntropy'] = val_metric_CrossEntropy
                    best_CrossEntropy = val_metric_CrossEntropy

                if val_metric_combined_kl_rmse < best_combined_kl_rmse:
                    if self._log:
                        wandb.run.summary['best_combined_kl_rmse'] = val_metric_combined_kl_rmse
                    best_combined_kl_rmse = val_metric_combined_kl_rmse

                if val_metric_kl < best_kl:
                    if self._log:
                        wandb.run.summary['best_kl'] = val_metric_kl
                    best_kl = val_metric_kl



                print("Validation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(self._loss, val_loss, 'target_nll',val_metric_nll, 'target_rmse', val_metric_rmse , 'target_mae', val_metric_mae, 'target_CrossEntropy', val_metric_CrossEntropy , 'target_combined_kl_rmse',val_metric_combined_kl_rmse , 'target_kl', val_metric_kl,'target_packet_rmse', val_metric_packet_rmse  ))

                if self._log:
                    wandb.log({self._loss + "/val_loss": val_loss, "nll/test_metric": val_metric_nll, "rmse/test_metric": val_metric_rmse, "packet_rmse/test_metric": val_metric_packet_rmse, "mae/test_metric": val_metric_mae, "CrossEntropy/test_metric": val_metric_CrossEntropy, "combined_kl_rmse/test_metric":val_metric_combined_kl_rmse, "epochs": i})

                if self._cluster_vis:
                    z_vis_concat = np.concatenate((z_vis, z_vis_val), axis=0)
                    z_labels_concat = np.concatenate((z_labels, z_labels_val), axis=0)
                    ####### Visualize the tsne embedding of the latent space in tensorboard
                    if self._log and i == (epochs - 1):
                        print('>>>>>>>>>>>>>Visualizing Latent Space<<<<<<<<<<<<<<', '---Epoch----', i)
                        ind = np.random.permutation(z_vis.shape[0])
                        z_vis = z_vis[ind, :]
                        z_vis = z_vis[:2000, :]
                        z_labels = z_labels[ind]
                        z_labels = z_labels[:2000]
                        ####### Visualize the tsne/pca in matplotlib / pyplot

                        plot_clustering(z_vis_concat, z_labels_concat, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)
                        plot_clustering(z_vis, z_labels, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)
                        plot_clustering(z_vis_val, z_labels_val, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)

        if self.c.learn.save_model:
            print("save last model")
            my_key = "3f542cd12f4a9ef15e8a6e681c56c9265cbee079"
            command = f'wandb login --relogin {my_key}'
            subprocess.run(command, shell=True, check=True)
            #name_artifact = str(self._run_name) +"_" + str(self._run_id)+"_model.ckpt"
            # now = datetime.datetime.now()
            # current_time = now.strftime("%H:%M:%S")
            #name_artifact = str(current_time)
            #print("name_artifact ", name_artifact)
            #print("line 747 trainer.py  self._save_path_local ", self._save_path_local)
            artifact.add_file(self._save_path_local)
            #local_path (required): The local path of the file that you want to add to the artifact.
            #name (optional): The name of the file inside the artifact. If not provided, the original filename will be used. ====> it seems that it is not very important for downloading the model later on. so I didnt specify it
            wandb.log_artifact(artifact)






        if self.c.learn.plot_traj:
            pass
  #           print('plotimp() from hiprssm_dyn_trainer.py line 416')
  #
  #           if(self._tar_type=="delta"):
  #               print("plotImputation with delta")      #                                                                                                                                        #the lasst 2 param are used to convert from diff to actual
  #               plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu, self._tr_sample_pred_var,self._run, log_name='train', exp_name=self._exp_name ,tar_type='delta',tar_obs=self._tr_sample_tar_obs ,normalizer=self._normalizer)
  #               plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu, self._te_sample_pred_var,self._run, log_name='test',  exp_name=self._exp_name, tar_type='delta',tar_obs=self._te_sample_tar_obs, normalizer=self._normalizer)
  #
  #
  #           if (self._tar_type == "observations" and self._loss!="cross_entropy"  and self._loss!="kl_rmse"):
  #               print("plotImputation with observation")
  #
  #               plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu,self._tr_sample_pred_var, self._run, log_name='train', exp_name=self._exp_name,tar_type="observations", tar_obs=self._tr_sample_tar_obs,normalizer=None)
  #               plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu,self._te_sample_pred_var, self._run, log_name='test', exp_name=self._exp_name ,tar_type="observations" , tar_obs=self._te_sample_tar_obs,normalizer=None)
  #