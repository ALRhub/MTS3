python experiments/mobileRobot/mts3_exp.py model=default_mts3 model.wandb.project_name='Test-Mobile-NewCode' model.wandb.exp_name="MTS3" --multirun &
python experiments/mobileRobot/acrkn_exp.py model=default_acrkn model.wandb.project_name='Test-Mobile-NewCode' model.wandb.exp_name="acRKN" --multirun &
python experiments/mobileRobot/hiprssm_exp.py model=default_hiprssm model.wandb.project_name='Test-Mobile-NewCode' model.wandb.exp_name="HiP-RSSM" --multirun &
python experiments/mobileRobot/rnn_exp.py model=default_lstm model.wandb.project_name='Test-Mobile-NewCode' model.wandb.exp_name="LSTM" --multirun &
python experiments/mobileRobot/rnn_exp.py model=default_gru model.wandb.project_name='Test-Mobile-NewCode' model.wandb.exp_name="GRU" --multirun
