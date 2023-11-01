
import sys
sys.path.append('.')
from omegaconf import DictConfig, OmegaConf
import hydra
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #import it before torch!  https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018/11
from hydra.utils import get_original_cwd
import subprocess


import numpy as np
import torch

import pickle

# from dataFolder.mobileDataSeq import metaMobileData
# from dataFolder.mobileDataSeq_Infer import metaMobileDataInfer
# from meta_dynamic_models.neural_process_dynamics.neural_process.setFunctionContext import SetEncoder
# from meta_dynamic_models.neural_process_dynamics.neural_process_ssm.recurrentEncoderDecoder import acrknContextualDecoder
# from meta_dynamic_models.neural_process_dynamics.npDynamics import npDyn
from learning import transformer_trainer #hiprssm_dyn_trainer
from inference import transformer_inference  # hiprssm_dyn_inference
from utils.metrics import naive_baseline
from utils.dataProcess import split_k_m
from utils.metrics import root_mean_squared
from utils.latentVis import plot_clustering, plot_clustering_1d
import random
from hydra.utils import get_original_cwd
import wandb


from transformer_architecture.model_transformer_TS import  TransformerModel, GPTConfig #--> auto-regressive model
#from transformer_architecture.models.Transformer_Longterm import  LongTermModel, TSConfig # ---> pred@once model

nn = torch.nn
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')






### sent by vaishak:

def reshape_data(data):
    ## reshape the dataFolder by flattening the second and third dimension
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2], -1)
    return data

def generate_mobile_robot_data_set(data):
    train_windows, test_windows = data.train_windows, data.test_windows

    train_targets = reshape_data(train_windows['target'])
    test_targets = reshape_data(test_windows['target'])

    train_obs = reshape_data(train_windows['obs'])
    test_obs = reshape_data(test_windows['obs'])

    train_task_idx = train_windows['task_index']
    test_task_idx = test_windows['task_index']

    train_obs_valid = reshape_data(train_windows['obs_valid'])
    test_obs_valid = reshape_data(test_windows['obs_valid'])

    train_task_valid = train_windows['task_valid']
    test_task_valid = test_windows['task_valid']

    train_act = reshape_data(train_windows['act'])
    test_act = reshape_data(test_windows['act'])

    print("Fraction of Valid Train Observations:",
          np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape))
    print("Fraction of Valid Test Observations:",
          np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))

    # print("train_targets.shape",train_targets.shape)
    # print("test_targets.shape" ,test_targets.shape)
    # print("train_obs.shape = " ,train_obs.shape)
    # print("train_act.shape = " ,train_act.shape)

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(
        train_targets).float(), torch.from_numpy(train_obs_valid).bool(), torch.from_numpy(
        train_task_valid).bool(), torch.from_numpy(train_task_idx).float(), \
        torch.from_numpy(test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(
        test_targets).float(), torch.from_numpy(test_obs_valid).bool(), torch.from_numpy(
        test_task_valid).bool(), torch.from_numpy(test_task_idx).float()


@hydra.main(config_path='conf_saleh',config_name='config')
def my_app(cfg)->OmegaConf:
    global config
    model_cfg = cfg.model
    exp = Experiment(model_cfg)

class Experiment():
    def __init__(self, cfg):
        self.global_cfg = cfg
        self._experiment()


    def _experiment(self):
        """Data"""



        cfg = self.global_cfg
        #print("before   cfg.learn.use_cuda   in main")

        #print("after  cfg.learn.use_cuda   in main")

        exp_seed = cfg.data_reader.seed

        torch.manual_seed(exp_seed)
        random.seed(exp_seed)
        np.random.seed(exp_seed)


        torch.cuda.empty_cache()

        tar_type = cfg.data_reader.tar_type  # 'delta' - if to train on differences to current states
        # 'next_state' - if to trian directly on the  next states
        #use_gpu = cfg.data_reader.use_cuda
        use_gpu= False
        device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")



        ## LOAD THE DATA

        dataset_name = cfg.data_reader.dataset_name

        # print((os.path.join(get_original_cwd(), 'all_data','neurips')))
        # print(os.listdir(os.path.join(get_original_cwd(), 'all_data','neurips')))
        path = os.path.join(get_original_cwd(), 'all_data','neurips', dataset_name)
        if torch.cuda.device_count() > 0:
            size_full = True

        else: #cpu
            size_full = False
            if dataset_name=='excavatorData.pkl': #hydraulic and cpu -->fake dataset
                dataset_name='excavatorDataFake.pkl'
                path = os.path.join(get_original_cwd(), 'all_data', 'neurips', dataset_name)
        if(cfg.wandb.exp_name[:3]=='tmp' and torch.cuda.device_count() > 0 ): #tmp config on gpu for sanity check
            size_full = False

        print("size_full:",size_full)

        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
            print("Train Obs Shape", data_dict['train_obs'].shape)
            print("Train Act Shape", data_dict['train_act'].shape)
            print("Train Targets Shape", data_dict['train_targets'].shape)
            print("Test Obs Shape", data_dict['test_obs'].shape)
            print("Test Act Shape", data_dict['test_act'].shape)
            print("Test Targets Shape", data_dict['test_targets'].shape)
            print("Normalizer", data_dict['normalizer'])
            if(size_full):
                train_obs       = data_dict['train_obs']
                train_act       = data_dict['train_act']
                train_targets   = data_dict['train_targets']

                test_obs        = data_dict['test_obs']
                test_act        = data_dict['test_act']
                test_targets    = data_dict['test_targets']
                normalizer      = data_dict['normalizer']
            else:
                train_obs = data_dict['train_obs'][:10,:,:]
                train_act = data_dict['train_act'][:10,:,:]
                train_targets = data_dict['train_targets'][:10,:,:]

                test_obs = data_dict['test_obs'][:2,:,:]
                test_act = data_dict['test_act'][:2,:,:]
                test_targets = data_dict['test_targets'][:2,:,:]
                normalizer = data_dict['normalizer']
        #train_obs, train_act, train_targets, train_obs_valid, train_task_valid, train_task_idx, test_obs, test_act, test_targets, test_obs_valid, test_task_valid, test_task_idx = generate_mobile_robot_data_set(dataFolder)


        act_dim = train_act.shape[-1]

        ####
        impu  = cfg.data_reader.imp
        epoch = cfg.learn.epochs


        ##### Define WandB Stuffs
        expName = cfg.wandb.exp_name
        if cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"
        n_layer = cfg.transformer_conf.n_layer
        n_head  = cfg.transformer_conf.n_head
        n_embed = cfg.transformer_conf.n_embed
        #update
        if cfg.learn.loss=='nll':
            pred_var = True
        else:
            pred_var=False


        my_conf = GPTConfig(vocab_size_in=train_obs.shape[-1]+train_act.shape[-1], vocab_size_out=train_obs.shape[-1] ,block_size=cfg.data_reader.context_size+cfg.data_reader.target_size+cfg.data_reader.burn_in_size ,n_layer=n_layer ,n_head=n_head , n_embed=n_embed , head_size= n_embed//n_head ,pred_var=pred_var)   #
        print("Transformer_config:",my_conf)
        #m = LongTermModel(my_conf)
        m = TransformerModel(my_conf)  #AR model
        n_params = sum(p.numel() for p in m.parameters())
        param_txt = str(n_params/1e6)[:5] +"M" #number of parameters in Milions
        print(param_txt)
        print("number of parameters: %.2fM" % (n_params / 1e6))



        if torch.cuda.device_count() > 1 and use_gpu:
            print("We have available ", torch.cuda.device_count(), "GPUs!")
            parellel_net = nn.DataParallel(m, device_ids=[0, 1, 2, 3])
        elif (torch.cuda.device_count() ==1 and use_gpu):
            print('only 1 GPU is avalable!')
            parellel_net = m
        else: #cpu
            print('Train on CPU:')
            parellel_net = m

        #input        = input.to(0)
        #parallel_net = parellel_net.to(0)
        parallel_net = parellel_net.to(device)
        device = next(parallel_net.parameters()).device
        if device.type == 'cpu':
            print("Model is on CPU")
        else:
            print("Model is on", device)


        ## Initializing wandb object and sweep object
        #wandb.logout()
        #os.environ["WANDB_API_KEY"] = "3f542cd12f4a9ef15e8a6e681c56c9265cbee079"
        my_key = "3f542cd12f4a9ef15e8a6e681c56c9265cbee079"
        command = f'wandb login --relogin {my_key}'
        subprocess.run(command, shell=True, check=True)

        #wandb.login(key=my_key , relogin=True)
        entity_name = "salehgh"

        #update in the other proj
        wandb_run = wandb.init( entity=entity_name   ,project=cfg.wandb.project_name, name=expName+"_epcoh"+str(cfg.learn.epochs)  +"_Loss-"+str(cfg.learn.loss)  +"_l-h-e_"+ str(n_layer)+"-"+str(n_head)+"-"+str(n_embed) +"_"+param_txt+"_Data-"+ dataset_name[:-4]+"_seed"+str(exp_seed) ,mode=mode)  # wandb object has a set of configs associated with it as well

        #is used for inference// it shows the LOCAL ADDRESS where to load the model from for the inference
        load_path= os.path.join( get_original_cwd() , 'experiments/saved_models' , str(wandb_run.name) , str(wandb_run.id) + '.ckpt' )
        print("load_path: ", load_path)



        print("**************************************************************************************")
        print("* line 143 mobile_robot_hiprssm.py   train_targets.shape = ",train_targets.shape,"*") #[x,y,1]
        print("* line 144 mobile_robot_hiprssm.py   train_obs.shape = ",train_obs.shape,"*")           #[q,w,#num_features]
        print("* line 145 mobile_robot_hiprssm.py   test_obs.shape = ", test_obs.shape),"*"
        print("* line 146 mobile_robot_hiprssm.py   test_targets.shape = ", test_targets.shape,"*")
        print("**************************************************************************************")


        transformer_learn = transformer_trainer.Learn(parallel_net, loss=cfg.learn.loss, imp=impu, config=cfg, run=wandb_run,log=cfg.wandb['log'],normalizer=normalizer )

        if cfg.learn.load == False:
            #### Train the Model
            print("..............Training the model..............")
            transformer_learn.train(train_obs, train_act, train_targets, cfg.learn.epochs, cfg.learn.batch_size, test_obs, test_act,test_targets,cfg=cfg)

        #sys.exit(1)

        if not cfg.wandb.sweep:
            #relogin to wandb in case in a shared cluster sb changed the wandb account
            my_key = "3f542cd12f4a9ef15e8a6e681c56c9265cbee079"
            command = f'wandb login --relogin {my_key}'
            subprocess.run(command, shell=True, check=True)
            ##### Load best model

            # artifact_name = str(wandb_run.name) +"_"+ str(wandb_run.id)  # the same as the one in trainer when we create and save the artifact
            # print("line 244 hiprssm.py artifact_name:",artifact_name)
            # model_at = wandb_run.use_artifact(artifact_name + ':latest')
            # model_path = model_at.download()  ###return the save directory path in wandb local
            # print("model_path = " ,model_path) # ./artifacts/saved_model:v39
            parallel_net.load_state_dict(torch.load(load_path))
            print('>>>>>>>>>>Loaded The Model From Local Folder<<<<<<<<<<<<<<<<<<<')

            transformer_infer = transformer_inference.Infer(parallel_net, data=None, config=cfg, run=wandb_run)
            batch_size = cfg.data_reader.multistep_batch_size #250
            k = cfg.data_reader.context_size  #=context_size=75



            #sys.exit(2)

            print("Multistep Test started......")
            multiSteps = [cfg.data_reader.target_size]
            for step in multiSteps:
                print("multisteps: line 142 mobile_robot_hiprssm.py",step,"/",multiSteps)

                pred_logits,  _ ,     gt_multi ,observed_part,pred_only_ctx = transformer_infer.predict_longterm(test_obs[:,:,:],  test_act[:,:,:],  test_targets[:,:,:], k=k,batch_size=batch_size,multiStep=step, tar=tar_type)  # returns normalized predicted packets
                if(cfg.learn.loss=='nll'):
                    B,T,C = observed_part.shape
                    B2,T2,C2 = pred_logits.shape
                    print("B,T,C =",B,T,C)
                    print("B2,T2,C2 =",B2,T2,C2)
                    #everything here is normalized

                    pred_mean = pred_logits[:,:,:C]
                    pred_var  = pred_logits[:,:,C:]

                    pred_only_ctx_mean = pred_only_ctx[:,:,:C]
                    pred_only_ctx_var  = pred_only_ctx[:,:,C:]
                if(cfg.learn.loss=='mse'):
                    pred_mean = pred_logits[:, :, :]
                    pred_only_ctx_mean = pred_only_ctx
                    print("hiprssm.py line 298: inference  pred_mean.shape= ",         pred_mean.shape)
                    print("hiprssm.py line 298: inference  pred_only_ctx_mean.shape ", pred_only_ctx_mean.shape)

                print("mobile_robot_hiprssm.py line 239 multstep, step=", step, "  pred_mean.shape =",pred_mean.shape , "  gt_multi.shape =",gt_multi.shape )  #pred_mean.shape = torch.Size([500, 750, 9])   gt_multi.shape = torch.Size([500, 750, 9])

                rmse_next_state_normalized, pred_obs, gt_obs             = root_mean_squared(pred_mean[:,:,:], gt_multi[:,:,:], normalizer=None, denorma=False,plot=True, steps=step,WB=wandb_run)  # step only used in the title of the plots
                rmse_next_state_denormalized, pred_obs_den, gt_obs_den   = root_mean_squared(pred_mean[:,:,:], gt_multi[:,:,:], normalizer=normalizer, denorma=True,plot=True, steps=step,WB=wandb_run)  # step only used in the title of the plots


                key1 = "Multistep_ERR_" + "Denormalized_" + str(step) + "step"
                wandb_run.summary[key1] = rmse_next_state_denormalized

                key2 = "Multistep_ERR_" + "Normalized_" + str(step) + "step"
                wandb_run.summary[key2] = rmse_next_state_normalized

                print('saving the GT and predicted array ....')

                spec = str(wandb_run.name)+"_id-"+str(wandb_run.id)


                np.savez(spec + "_norm_pred_mean.npz"    , data=pred_mean)    #both: ctx[:,1:,:] and target[:,:,:]
                np.savez(spec + "_denorm_pred_mean.npz"  , data=pred_obs_den) #both: ctx[:,1:,:] and target[:,:,:]
                np.savez(spec + "_norm_gt.npz"      , data=gt_multi)     # = torch.cat([ctx_obs_batch, tar_obs_batch], dim=1)[:,1:,:]
                np.savez(spec + "_denorm_gt.npz"    , data=gt_obs_den)   # = torch.cat([ctx_obs_batch, tar_obs_batch], dim=1)[:,1:,:]
                np.savez(spec + "_normalizer.npz"   , data=normalizer)
                np.savez(spec + "_observed_part.npz", data=observed_part) #torch.cat([ctx_inp, tar_inp[:,:tar_burn_in,:]], dim=1) [:,:,:obs_dim] # first sample of obs only can be seen here

                np.savez(spec + "_pred_only_ctx_mean.npz", data=pred_only_ctx_mean.cpu().numpy())


                if(cfg.learn.loss=='nll'): # predicted variances on ctx and target
                    np.savez(spec + "_var_pred.npz", data=pred_var.cpu().numpy())
                    np.savez(spec + "_pred_only_ctx_var.npz", data=pred_only_ctx_var.cpu().numpy())


                ## to laod:
                #tmp = np.load("file-name.npz")
                #norm_pred = tmp['data']




def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()