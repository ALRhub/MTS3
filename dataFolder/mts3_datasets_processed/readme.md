The datasets can be downloaded from the following urls. Its automatically downloaded when running exps, if not
already present.

Mobile Robot: https://bwsyncandshare.kit.edu/s/nNkQfMGtZgkGjJJ/download

Maze2d(medium): https://bwsyncandshare.kit.edu/s/SmRYcDj9HnSx97r/download

HalfCheetah: https://bwsyncandshare.kit.edu/s/LY662F3yxWxA99Z/download

FrankaKitchen: https://bwsyncandshare.kit.edu/s/Hnma3nj47NnJsEs/download

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