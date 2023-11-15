# TODO: collect valid flags too ??
# TODO: go through the code once again
# TODO: check if update and marginalization is correct
import torch
from agent.worldModels.SensorEncoders.propEncoder import Encoder
from agent.worldModels.gaussianTransformations.gaussian_marginalization import Predict
from agent.worldModels.gaussianTransformations.gaussian_conditioning import Update
from agent.worldModels.Decoders.propDecoder import SplitDiagGaussianDecoder

nn = torch.nn

'''
Tip: in config self.ltd = lod 
in context_predict ltd is doubled
in task_predict ltd is considered lod and lsd is doubled inside
'''
class MTS3Simple(nn.Module):
    """
    MTS3 model
    Inference happen in such a way that first episode is used for getting an intial task posterioer and then the rest of the episodes are used for prediction by the worker
    Maybe redo this logic based on original implementation or use a different method that helps control too ??
    """

    def __init__(self, input_shape=None, config=None, use_cuda_if_available: bool = True):
        """
        @param obs_dim: dimension of observations to train on
        @param inp_shape: shape of the input observations
        @param config: config dict
        @param dtype:
        @param use_cuda_if_available:
        """
        super(MTS3Simple, self).__init__()
        if config == None:
            raise ValueError("config cannot be None, pass an omegaConf File")
        else:
            self.c = config
        self.H = self.c.mts3.time_scale_multiplier
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._obs_shape = input_shape
        self._lod = self.c.mts3.latent_obs_dim
        self._lsd = 2*self._lod
        self._time_embed_dim = self.c.mts3.manager.abstract_obs_encoder.time_embed.dim
        assert self._time_embed_dim == self.c.mts3.manager.abstract_act_encoder.time_embed.dim, \
                                            "Time Embedding Dimensions for obs and act encoder should be same"
        self._pixel_obs = self.c.mts3.pixel_obs ##TODO: config
        self._decode_obs = self.c.mts3.decode.obs ##TODO: config and it basically initializes the obs decoder



        ### Define the encoder and decoder
        obsEnc = Encoder(self._obs_shape[-1], self._lod, self.c.mts3.worker.obs_encoder) ## TODO: config
        self._obsEnc = obsEnc.to(self._device)

        absObsEnc = Encoder(self._obs_shape[-1] + self._time_embed_dim, self._lod, self.c.mts3.manager.abstract_obs_encoder) ## TODO: config
        self._absObsEnc = absObsEnc.to(self._device)

        obsDec = SplitDiagGaussianDecoder(latent_obs_dim=self._lod, out_dim=self._obs_shape[-1], config=self.c.mts3.worker.obs_decoder) ## TODO: config
        self._obsDec = obsDec.to(self._device)


        ### Define the gaussian layers for both levels
        ### Since unactuated dynamics, the action dimension is None
        self._state_predict = Predict(latent_obs_dim=self._lod, act_dim=None, hierarchy_type = "worker", config=self.c.mts3.worker).to(self._device) 
                                                                                    ## initiate worker marginalization layer for state prediction
        self._task_predict = Predict(latent_obs_dim=self._lod, act_dim=None, hierarchy_type = "manager", config=self.c.mts3.manager).to(self._device)
                                                                                    ## initiate manager marginalization layer for task prediction

        self._obsUpdate = Update(latent_obs_dim=self._lod, memory = True, config = self.c).to(self._device) ## memory is true
        self._taskUpdate = Update(latent_obs_dim=self._lod, memory = True, config = self.c).to(self._device) ## memory is true

    def _intialize_mean_covar(self, batch_size, learn=False):
        if learn:
            pass

        else:
            init_state_covar_ul = self.c.mts3.initial_state_covar * torch.ones(batch_size, self._lsd)

            initial_mean = torch.zeros(batch_size, self._lsd).to(self._device)
            icu = init_state_covar_ul[:, :self._lod].to(self._device)
            icl = init_state_covar_ul[:, self._lod:].to(self._device)
            ics = torch.ones(batch_size, self._lod).to(self._device)

            initial_cov = [icu, icl, ics]

        return initial_mean, initial_cov 


    def _create_time_embedding(self, batch_size, time_steps):
        """
        Creates a time embedding for the given batch size and time steps
        of the form (batch_size, time_steps, 1)"""
        time_embedding = torch.zeros(batch_size, time_steps, 1).to(self._device)
        for i in range(time_steps):
            time_embedding[:, i, :] = i / time_steps
        return time_embedding
    
    def _pack_variances(self, variances):
        """
        pack list of variances (upper, lower, side) into a single tensor
        """
        return torch.cat(variances, dim=-1)
    
    def _unpack_variances(self, variances):
        """
        unpack list of variances (upper, lower, side) from a single tensor
        """
        return torch.split(variances, self._lod, dim=-1)
    
    def forward(self, obs_seqs, obs_valid_seqs, train=False):
        '''
        obs_seqs: sequences of timeseries of observations (batch x time x obs_dim)
        action_seqs: sequences of timeseries of actions (batch x time x obs_dim)
        obs_valid_seqs: sequences of timeseries of actions (batch x time)
        '''
        ##################################### Manager ############################################
        # prepare list for return
        prior_task_mean_list = []
        prior_task_cov_list = []

        post_task_mean_list = []
        post_task_cov_list = []

        ### initialize mean and covariance for the first time step
        task_prior_mean_init, task_prior_cov_init = self._intialize_mean_covar(obs_seqs.shape[0], learn=False)

        ### loop over individual episodes in steps of H (Coarse time scale / manager)
        for k in range(0, obs_seqs.shape[1], self.H):
            #print("Episode: ", k)
            episode_num = int(k // self.H)
            if k==0:
                task_prior_mean = task_prior_mean_init
                task_prior_cov = task_prior_cov_init
            ### encode the observation set with time embedding to get abstract observation
            current_obs_seqs = obs_seqs[:, k:k+self.H, :]
            time_embedding = self._create_time_embedding(current_obs_seqs.shape[0], current_obs_seqs.shape[1]) 
                                                                        # [x] made sure works with episodes < H
            beta_k_mean, beta_k_var = self._absObsEnc(torch.cat([current_obs_seqs, time_embedding], dim=-1))

            ### get task valid for the current episode
            obs_valid = obs_valid_seqs[:, k:k+self.H, :] 
                        #[x] created in learn class, with interwindow (whole windows masked) and intrawindow masking
            ### update the task posterior with beta_current
            task_post_mean, task_post_cov = self._taskUpdate(task_prior_mean, task_prior_cov, beta_k_mean, beta_k_var, obs_valid)
            

            ### predict the next task mean and covariance using manager marginalization layer 
            ### with the current task posterior as causal factor
            mean_list_causal_factors = [task_post_mean] 
            cov_list_causal_factors = [task_post_cov]
            task_next_mean, task_next_cov = self._task_predict(mean_list_causal_factors, cov_list_causal_factors) #[.]: absact inference some problem fixed.
            
            ### update the task prior
            task_prior_mean, task_prior_cov = task_next_mean, task_next_cov

            ### append the task mean and covariance to the list
            prior_task_mean_list.append(task_prior_mean)
            prior_task_cov_list.append(self._pack_variances(task_prior_cov)) ## append the packed covariances
            post_task_mean_list.append(task_post_mean)
            post_task_cov_list.append(self._pack_variances(task_post_cov)) ## append the packed covariances
            

        ### stack the list to get the final tensors
        prior_task_means = torch.stack(prior_task_mean_list, dim=1)
        prior_task_covs = torch.stack(prior_task_cov_list, dim=1)
        post_task_means = torch.stack(post_task_mean_list, dim=1)
        post_task_covs = torch.stack(post_task_cov_list, dim=1)

        ### get the number of episodes from the length of prior_task_mean
        num_episodes = prior_task_means.shape[1] 

        
        ##################################### Worker ############################################
        ### using the task prior, predict the observation mean and covariance for fine time scale / worker
        ### create a meta_list of prior and posterior states

        global_state_prior_mean_list = []
        global_state_prior_cov_list = []
        global_state_post_mean_list = []
        global_state_post_cov_list = []

        state_prior_mean_init, state_prior_cov_init = self._intialize_mean_covar(obs_seqs.shape[0], learn=False)

        for k in range(0,num_episodes): ## first episode is considered too to predict (but usually ignored in evaluation)
            #print("Episode: ", k)
            if k==0:
                state_prior_mean = state_prior_mean_init
                state_prior_cov = state_prior_cov_init
            ### create list of state mean and covariance 
            prior_state_mean_list = []
            prior_state_cov_list = []
            post_state_mean_list = []
            post_state_cov_list = []

            ### get the task post for the current episode (here the assumption is that when observations are missing the task valid flag keeps posterior = prior)
            task_mean = post_task_means[:, k, :]
            task_cov = self._unpack_variances(post_task_covs[:, k, :])

            ### get the obs, action for the current episode
            current_obs_seqs = obs_seqs[:, k*self.H:(k+1)*self.H, :]
            current_obs_valid_seqs = obs_valid_seqs[:, k*self.H:(k+1)*self.H, :]
            current_episode_len = current_obs_seqs.shape[1] # [x] made sure works with episodes < H

            for t in range(current_episode_len): # [x] made sure works with episodes < H
                #print("Time Step: ", t)
                ### encode the observation (no time embedding)
                current_obs = current_obs_seqs[:, t, :]
                obs_mean, obs_var = self._obsEnc(current_obs)

                ## expand dims to make it compatible with the update step (which expects a 3D tensor)
                obs_mean = torch.unsqueeze(obs_mean, dim=1)
                obs_var = torch.unsqueeze(obs_var, dim=1)

                ### update the state posterior
                current_obs_valid = current_obs_valid_seqs[:, t, :]
                ## expand dims to make it compatible with the encoder
                current_obs_valid = torch.unsqueeze(current_obs_valid, dim=1)
                state_post_mean, state_post_cov = self._obsUpdate(state_prior_mean, state_prior_cov, obs_mean, obs_var, current_obs_valid)
                #state_post_mean, state_post_cov = state_prior_mean, state_prior_cov ##TODO: remove this line and uncomment the above line

                ### predict the next state mean and covariance using the marginalization layer for worker
                mean_list_causal_factors = [state_post_mean, task_mean]
                cov_list_causal_factors = [state_post_cov, task_cov]
                
                state_next_mean, state_next_cov = self._state_predict(mean_list_causal_factors, cov_list_causal_factors)

                ### update the state prior
                state_prior_mean, state_prior_cov = state_next_mean, state_next_cov  ### this step also makes sure every episode 
                                                                                        ### starts with the prior of the previous episode

                ### concat 
                ### append the state mean and covariance to the list
                prior_state_mean_list.append(state_prior_mean)
                prior_state_cov_list.append(torch.cat(state_prior_cov, dim=-1))
                post_state_mean_list.append(state_post_mean)
                post_state_cov_list.append(torch.cat(state_post_cov, dim=-1))
            
            ## detach the state prior mean and covariance to make sure the next episode starts with the prior of the previous episode
            state_prior_mean = state_prior_mean.detach()
            state_prior_cov = [cov.detach() for cov in state_prior_cov]

            ### stack the list to get the final tensors
            prior_state_means = torch.stack(prior_state_mean_list, dim=1)
            prior_state_covs = torch.stack(prior_state_cov_list, dim=1)
            post_state_means = torch.stack(post_state_mean_list, dim=1)
            post_state_covs = torch.stack(post_state_cov_list, dim=1)

            ### append the state mean and covariance to the list
            global_state_prior_mean_list.append(prior_state_means) 
            global_state_prior_cov_list.append(prior_state_covs)
            global_state_post_mean_list.append(post_state_means)
            global_state_post_cov_list.append(post_state_covs)

        ### concat along the episode dimension
        global_state_prior_means = torch.cat(global_state_prior_mean_list, dim=1)
        global_state_prior_covs = torch.cat(global_state_prior_cov_list, dim=1)
        global_state_post_means = torch.cat(global_state_post_mean_list, dim=1)
        global_state_post_covs = torch.cat(global_state_post_cov_list, dim=1)

        ##################################### Decoder ############################################
        ### decode the state to get the observation mean and covariance ##TODO: do it here ?? or outside ???
        if self._decode_obs:
            pred_obs_means, pred_obs_covs = self._obsDec(global_state_prior_means, global_state_prior_covs)
            
        return pred_obs_means, pred_obs_covs, prior_task_means.detach(), prior_task_covs.detach(), post_task_means.detach(), post_task_covs.detach()
