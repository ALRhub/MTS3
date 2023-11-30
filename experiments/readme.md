## Config Structure For Experiments

Each baselines hyperparameters can be found in the configuration files (yaml files) in the conf folder under the corresponding dataset folder.

For example for the experiments with Mobile Robot dataset, the configuration files are:

    - experiments/mobileRobot/conf/model/default_mts3.yaml ##for mts3 model hyperparameters
    - experiments/mobileRobot/conf/model/learn/default.yaml ##for learning/training/optimizer related hyperparameters 

The configuration files for the other baselines and datasets can be found in the corresponding folders.

Refer to [hydra](https://hydra.cc/docs/intro/) documentation for more details on how to use the configuration files and how these are used in the script. However to just run the scripts you don't need to understand hydras configuration framework.

## Running Experiments with another hyperparameter configuration

1. You can either change the hyperparameters in the configuration files or create a new configuration file with the desired hyperparameters.
2. Then you can run the experiment with the new configuration file by passing the configuration file as an argument to the script. For example:

    ```python
    python experiments/mobileRobot/mts3_exp.py model=default_mts3 
    ```

    This will run the experiment with the hyperparameters specified in the configuration file `default_mts3.yaml` in the `conf/model` folder.
3. You can also pass the hyperparameters as command line arguments. For example:

    ```python
    python experiments/mobileRobot/mts3_exp.py model=default_mts3 model.encoder.hidden_size=256 model.learn.batch_size=500
    ```

    This will run the experiment with the hyperparameters specified in the configuration file `default_mts3.yaml` in the `conf/model` folder and will override the `encoder.hidden_size` parameter with the value `256`. It will also override the `learn.batch_size` parameter with the value `500`.