The datasets can be downloaded from the following urls. Its automatically downloaded when running exps, if not
already present.

Mobile Robot: https://drive.google.com/file/d/1ShZ_LyLGkWimboJv5RRk8M4yvrT7AUYY/view?usp=drive_link

Maze2d(medium): https://drive.google.com/file/d/1fwrLrV_28832OYat4YpWuNl51MEXbKlq/view?usp=drive_link

HalfCheetah: https://drive.google.com/file/d/1MuJBYSNN3D6BRfGp0Eu7Hbz1roKOzcxN/view?usp=drive_link

FrankaKitchen: https://drive.google.com/file/d/1DDUpJdHUec_4WsMgO9B7bpMSkbhSwV1X/view?usp=drive_link

**<span style="color:red;">Important Note:</span>** All datasets are normalized to mean zero std one. We also provide the normalization constants with the datasets. The normalization constants are stored in the `normalizer` key in the data dictionary.

Use the following script to load the data.

```python
import pickle
##print all shapes
with open(data_path, 'rb') as f:     
          data_dict = pickle.load(f)     
          print("Train Obs Shape", data_dict['train_obs'].shape) 
          print("Train Act Shape", data_dict['train_act'].shape)
          print("Train Targets Shape", data_dict['train_targets'].shape)
          print("Test Obs Shape", data_dict['test_obs'].shape) 
          print("Test Act Shape", data_dict['test_act'].shape)
          print("Test Targets Shape", data_dict['test_targets'].shape) 
          print("Normalizer", data_dict['normalizer'])
```