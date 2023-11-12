The datasets can be downloaded from the following urls. Its automatically downloaded when running exps, if not
already present.

Mobile Robot: https://bwsyncandshare.kit.edu/s/nNkQfMGtZgkGjJJ/download

Maze2d(medium): https://bwsyncandshare.kit.edu/s/SmRYcDj9HnSx97r/download

HalfCheetah: https://bwsyncandshare.kit.edu/s/LY662F3yxWxA99Z/download

FrankaKitchen: https://bwsyncandshare.kit.edu/s/Hnma3nj47NnJsEs/download

All datasets are normalized to mean zero std one.
Use the following script to load them.

```python
with open(data_path, 'wb') as f:     
          pickle.dump(data_dict, f)#### Load json data and
                                                ##print all shapes
with open(get_original_cwd() + data_path, 'rb') as f:     
          data_dict = pickle.load(f)     
          print("Train Obs Shape", data_dict['train_obs'].shape) 
          print("Train Act Shape", data_dict['train_act'].shape)
          print("Train Targets Shape", data_dict['train_targets'].shape)
          print("Test Obs Shape", data_dict['test_obs'].shape) 
          print("Test Act Shape", data_dict['test_act'].shape)
          print("Test Targets Shape", data_dict['test_targets'].shape) 
          print("Normalizer", data_dict['normalizer'])
```
