import torch
import numpy as np


def split_k_m(sequence,k,burn_in=5):
    '''
    TODO: K and M as argument instead of split by half
    '''
    if k==0:
        context_seq, target_seq = None, sequence
    else:
        context_seq, target_seq = sequence[:, :2*k, :], sequence[:, k-burn_in:, :]
    return context_seq, target_seq

def get_ctx_target_impute(obs, act, target, k, num_context=None, test_gt_known=True, tar_imp=0.0, ctx_burn_in=0, tar_burn_in=5, mix=True, multistep=False, random_seed=True):
    '''
    :param obs: observations
    :param act: actions
    :param target: targets
    :param k: how many timesteps to have for context sequence after splitting the sequence
    :param num_context: if we want a context size less than k, in paper None by default
    :param test_gt_known: is test ground truth available
    :param tar_imp: percentage imputation in target sequence
    :param ctx_burn_in: how much context to burnin
    :param tar_burn_in: how much of target should be used as burnin
    :param random_seed:
    :return:
    '''
    if random_seed:
        seed = np.random.randint(1, 1000)
    else:
        seed = 42

    if ctx_burn_in is None:
        ctx_burn_in = k

    rs = np.random.RandomState(seed=seed)
    ctx_obs, tar_obs = split_k_m(obs, k, burn_in=ctx_burn_in)
    ctx_act, tar_act = split_k_m(act, k, burn_in=ctx_burn_in)
    if test_gt_known:
        ctx_tar, tar_tar = split_k_m(target, k, burn_in=ctx_burn_in)
    else:
        ctx_tar, tar_tar = target, None
    # Sample locations of context points
    if num_context is not None:
        num_context = num_context
        locations = np.random.choice(k,
                                     size=num_context,
                                     replace=False)
        #TODO: change this randomization per episode too
        ctx_obs = ctx_obs[:, locations[:k], :]
        ctx_act = ctx_act[:, locations[:k], :]
        ctx_tar = ctx_tar[:, locations[:k], :]

    if mix:
        impu = 1 - np.random.uniform(0,tar_imp)
    else:
        impu = 1 - tar_imp


    if tar_imp is not None:
        tar_obs_valid = rs.rand(tar_obs.shape[0], tar_obs.shape[1], 1) < impu
        tar_obs_valid[:, :tar_burn_in] = True
        ## from step to rest set to False
        if multistep:
            tar_obs_valid[:, k:] = False
    else:
        tar_obs_valid = rs.rand(tar_obs.shape[0], tar_obs.shape[1], 1) < 0
        tar_obs_valid[:, :tar_burn_in] = True
        if multistep:
            tar_obs_valid[:, k:] = False
    return ctx_obs, ctx_act, ctx_tar, tar_obs, tar_act, tar_tar, tar_obs_valid

def get_ctx_target_multistep(obs, act, target, k, steps=1, num_context=None, test_gt_known=True, ctx_burn_in=0, tar_burn_in=5, random_seed=True):
    '''
    :param obs: observations
    :param act: actions
    :param target: targets
    :param k: how many timesteps to have for context sequence after splitting the sequence
    :param num_context: if we want a context size less than k, in paper None by default
    :param test_gt_known: is test ground truth available
    :param tar_imp: percentage imputation in target sequence
    :param ctx_burn_in: how much context to burnin
    :param tar_burn_in: how much of target should be used as burnin
    :param random_seed:
    :return:
    '''
    if random_seed:
        seed = np.random.randint(1, 1000)
    else:
        seed = 42

    if ctx_burn_in is None:
        ctx_burn_in = k

    rs = np.random.RandomState(seed=seed)
    ctx_obs, tar_obs = split_k_m(obs, k, burn_in=ctx_burn_in) # split the obs sequence into context and target
    ctx_act, tar_act = split_k_m(act, k, burn_in=ctx_burn_in) # split the act sequence into context and target

    if test_gt_known:
        ctx_tar, tar_tar = split_k_m(target, k, burn_in=ctx_burn_in) # split the target sequence into context and target
    else:
        ctx_tar = target
        tar_tar = None

    # Sample locations of context points
    if num_context is not None:
        num_context = num_context
        locations = np.random.choice(k,
                                     size=num_context,
                                     replace=False)
        #TODO: change this randomization per episode too and add test_gt_stuff
        ctx_obs = ctx_obs[:, locations[:k], :]
        ctx_act = ctx_act[:, locations[:k], :]
        ctx_tar = ctx_tar[:, locations[:k], :]

    # create valid flags for target sequence
    tar_obs_valid = rs.rand(tar_obs.shape[0], tar_obs.shape[1], 1) < 1 #Everything True
    tar_obs_valid[:, -steps:] = False #Last steps are set to False
    tar_obs_valid = torch.from_numpy(tar_obs_valid).bool() #Convert to torch tensor and bool


    return ctx_obs, ctx_act, ctx_tar, tar_obs, tar_act, tar_tar, tar_obs_valid

def get_sliding_context_batch_mbrl(obs, act, target, k, steps=1, tar_burn_in=5):
    '''
    Given say N episodes, it creates context/target windows in the ration k:(step+tar_burn_in).
    The window centers are ordered as opposed to random.
    :param obs:
    :param act:
    :param target:
    :param k: context size
    :param steps: multi step ahead prediction to make
    :return:
    '''
    #print(data_out.shape)
    tar_length = steps + tar_burn_in
    H = obs.shape[1]
    #Creating ordered window centres
    window_centres = np.arange(k,H-tar_length+1) #the +1 is very important
    #Creating windows starting from a particular context in a sliding window fashion
    obs_hyper_batch = [obs[:, ind - k:ind + tar_length, :] for ind in window_centres]
    act_hyper_batch = [act[:, ind - k:ind + tar_length, :] for ind in window_centres]
    target_hyper_batch = [target[:, ind - k:ind + tar_length, :] for ind in window_centres]

    return torch.cat(obs_hyper_batch,dim=0), torch.cat(act_hyper_batch,dim=0), torch.cat(target_hyper_batch, dim=0)

def _create_valid_flags_multistep(self,obs, steps):
    rs = np.random.RandomState(seed=42)
    obs_valid_batch = rs.rand(obs.shape[0], obs.shape[1], obs.shape[2], 1) < 1

    task_valid_batch = rs.rand(obs.shape[0], obs.shape[1], 1) < 1
    task_valid_batch[:, -steps:] = False


    return torch.from_numpy(obs_valid_batch).bool(), torch.from_numpy(task_valid_batch).bool()

def get_sliding_dpssm_batch_mbrl(obs, act, target, steps=1, task_burn_in=1):
    '''
    Given say N episodes, it creates context/target windows in the ration k:(step+tar_burn_in).
    The window centers are ordered as opposed to random.
    :param obs:
    :param act:
    :param target:
    :param k: context size
    :param steps: multi step ahead prediction to make
    :return:
    '''
    #TODO: Dont use... Not proper
    #print(data_out.shape)
    tar_length = steps + task_burn_in
    H = obs.shape[1]
    #Creating ordered window centres
    window_end_pos = np.arange(tar_length,H) #the +1 is very important
    #Creating windows starting from a particular context in a sliding window fashion
    obs_hyper_batch = torch.cat([obs[:, :ind+1, :, :] for ind in window_end_pos],dim=0)
    act_hyper_batch = torch.cat([act[:, :ind+1, :, :] for ind in window_end_pos],dim=0)
    target_hyper_batch = torch.cat([target[:, :ind+1, :, :] for ind in window_end_pos],dim=0)

    return obs_hyper_batch, act_hyper_batch, target_hyper_batch

def squeeze_sw_batch(pred_mean_hyper_batch, pred_var_hyper_batch, target_hyper_batch, num_episodes):
    '''
    :param pred_hyper_batch:
    :param target_hyper_batch:
    :return: predicted and ground truth sequence and has number of episodes = num_episodes
    '''
    if type(pred_mean_hyper_batch) is np.ndarray:
        pred_mean_hyper_batch = torch.from_numpy(pred_mean_hyper_batch).float()
    if type(pred_var_hyper_batch) is np.ndarray:
        pred_var_hyper_batch = torch.from_numpy(pred_var_hyper_batch).float()
    if type(target_hyper_batch) is np.ndarray:
        target_hyper_batch = torch.from_numpy(target_hyper_batch).float()
    hyper_episodes = pred_mean_hyper_batch.shape[0]
    hyper_windows_per_episode = int(hyper_episodes/num_episodes)
    assert num_episodes*hyper_windows_per_episode == hyper_episodes
    for ind in range(hyper_windows_per_episode):
        if ind==0:
            squeezed_pred_mean = pred_mean_hyper_batch[ind*num_episodes:(ind+1)*num_episodes,:,:]
            squeezed_pred_var = pred_var_hyper_batch[ind * num_episodes:(ind + 1) * num_episodes, :, :]
            squeezed_gt = target_hyper_batch[ind*num_episodes:(ind+1)*num_episodes,:,:]
        else:
            squeezed_pred_mean = torch.cat((squeezed_pred_mean,
                                           torch.unsqueeze(pred_mean_hyper_batch[ind * num_episodes:(ind + 1) * num_episodes, -1, :],dim=1)),
                                           dim=1)

            squeezed_pred_var = torch.cat((squeezed_pred_var,
                                           torch.unsqueeze(pred_var_hyper_batch[ind * num_episodes:(ind + 1) * num_episodes, -1, :], dim=1)),
                                           dim=1)

            squeezed_gt = torch.cat((squeezed_gt,torch.unsqueeze(target_hyper_batch[ind*num_episodes:(ind+1)*num_episodes,-1,:], dim=1)), dim=1)
    return squeezed_pred_mean, squeezed_pred_var, squeezed_gt





seqToArray = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
def arrayToSeq(x, numEp, epLen):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    return np.reshape(x, (numEp, epLen, -1))

def normalize(data, mean, std):
        dim = data.shape[-1]
        return (data - mean[:dim]) / (std[:dim] + 1e-10)

def denormalize(data, mean, std):
    dim = data.shape[-1]
    return data * (std[:dim] + 1e-10) + mean[:dim]

def norm(x,normalizer,tar_type='targets'):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    if tar_type=='observations':
        return normalize(x, normalizer["observations"][0][:x.shape[-1]],
                   normalizer["observations"][1][:x.shape[-1]])
    if tar_type == 'actions':
        return normalize(x, normalizer["actions"][0][:x.shape[-1]],
                              normalizer["actions"][1][:x.shape[-1]])

    else:
        return normalize(x, normalizer["targets"][0][:x.shape[-1]],
                       normalizer["targets"][1][:x.shape[-1]])

def denorm_var(x,normalizer,tar_type='targets'):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    if tar_type=='observations':
        return x * (normalizer["observations"][1][:x.shape[-1]] + 1e-10)**2
    if tar_type == 'actions':
        return x * (normalizer["actions"][1][:x.shape[-1]] + 1e-10)**2
    else:
        return x * (normalizer["targets"][1][:x.shape[-1]] + 1e-10)**2




def denorm(x, normalizer, tar_type='targets'):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    if tar_type=='observations':
        return denormalize(x, normalizer["observations"][0][:x.shape[-1]],
                   normalizer["observations"][1][:x.shape[-1]])
    if tar_type == 'actions':
        return denormalize(x, normalizer["actions"][0][:x.shape[-1]],
                                normalizer["actions"][1][:x.shape[-1]])
    if tar_type == 'act_diff':
        return denormalize(x, normalizer["act_diff"][0][:x.shape[-1]],
                                normalizer["act_diff"][1][:x.shape[-1]])


    else:
        return denormalize(x, normalizer["targets"][0][:x.shape[-1]],
                       normalizer["targets"][1][:x.shape[-1]])


def diffToState(diff,current,normalizer,standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state
    '''
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(current) is not np.ndarray:
        current = current.cpu().detach().numpy()

    if standardize:
        current = denorm(current, normalizer, 'observations')
        diff = denorm(diff, normalizer, "diff")

        next = norm(current + diff, normalizer, "observations")
    else:
        next = current + diff

    return next,diff

def diffToStateMultiStep(diff, current, valid_flag, normalizer, standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state, diff
    '''
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(current) is not np.ndarray:
        current = current.cpu().detach().numpy()



    if standardize:
        current = denorm(current, normalizer, 'observations')
        diff = denorm(diff, normalizer, "diff")

    next_state = np.zeros(current.shape)

    for t in range(current.shape[1]):
        '''
        Loop over the differences and check is valid flag is true or false
        '''
        if valid_flag[0,t,0] == False and t>0:
            next_state[:,t] = next_state[:,t-1] + diff[:,t]
        else:
            next_state[:,t] = current[:,t] + diff[:,t]

    next_norm =  norm(next_state, normalizer, 'observations')
    diff_norm =  norm(diff, normalizer, 'diff')
    return next_norm, diff_norm


def diffToStateImpute(diff, current, valid_flag, normalizer, standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state, diff
    '''
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(current) is not np.ndarray:
        current = current.cpu().detach().numpy()
    if type(valid_flag) is not np.ndarray:
        valid_flag = valid_flag.cpu().detach().numpy()

    if len(diff.shape) == 4:
        #TODO: check if this reshaping gets what you want
        print(diff.shape,current.shape,valid_flag.shape)
        diff = np.reshape(diff, (diff.shape[0],diff.shape[1]*diff.shape[2],-1))
        current = np.reshape(current, (current.shape[0], current.shape[1] * current.shape[2], -1))
        valid_flag = np.reshape(valid_flag, (valid_flag.shape[0], valid_flag.shape[1] * valid_flag.shape[2], 1))


    if standardize:
        current = denorm(current, normalizer, 'observations')
        diff = denorm(diff, normalizer, "diff")
    next_state = np.zeros(current.shape)

    for idx in range(current.shape[0]):
        for t in range(current.shape[1]):
            '''
            Loop over the differences and check is valid flag is true or false
            '''
            if valid_flag[idx, t, 0] == False and t > 0:
                #changed to idx, check if correct or not
                next_state[idx, t] = next_state[idx, t - 1] + diff[idx, t]
            else:
                next_state[idx, t] = current[idx, t] + diff[idx, t]

    next_norm = norm(next_state, normalizer, 'observations')
    diff_norm = norm(diff, normalizer, 'diff')
    return next_norm, diff_norm

def diffToAct(diff,prev,normalizer,standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state
    '''
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(prev) is not np.ndarray:
        prev = prev.cpu().detach().numpy()

    if standardize:
        prev = denorm(prev, normalizer, 'actions')
        diff = denorm(diff, normalizer, "act_diff")

        current = norm(prev + diff, normalizer, "actions")

    return current,diff



