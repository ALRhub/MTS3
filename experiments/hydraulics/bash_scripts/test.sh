python experiments/hydraulics/mts3_exp.py model=default_mts3 model.wandb.project_name='Test-Hydraulics-NewCode' model.wandb.exp_name="MTS3" --multirun &
python experiments/hydraulics/mts3_exp.py model=default_mts3_NoI model.wandb.project_name='Test-Hydraulics-NewCode' model.wandb.exp_name="MTS3-NoI" --multirun &
python experiments/hydraulics/acrkn_exp.py model=default_acrkn model.wandb.project_name='Test-Hydraulics-NewCode' model.wandb.exp_name="acRKN" --multirun &
python experiments/hydraulics/hiprssm_exp.py model=default_hiprssm model.wandb.project_name='Test-Hydraulics-NewCode' model.wandb.exp_name="HiP-RSSM" --multirun &
#python experiments/hydraulics/rnn_exp.py model=default_lstm model.wandb.project_name='Test-Hydraulics-NewCode' model.wandb.exp_name="LSTM" --multirun &
#python experiments/hydraulics/rnn_exp.py model=default_gru model.wandb.project_name='Test-Hydraulics-NewCode' model.wandb.exp_name="GRU" --multirun
