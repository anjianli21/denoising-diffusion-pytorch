## Diffusion model for trajectory optimization in CR3BP, in Pytorch

Classifier free 1D conditional diffusion model.

## Dataset configuration

Creat a folder "/data/CR3BP" and put the dataset "cr3bp_time_mass_alpha_control_part_4_250k_each.pkl" under this folder.

## Usage: training



```python
$ python run/train_classifier_free_cond_1d.py
```

## Usage: testing

First configure “results_folder" and "milestone" in "run/test_classifier_free_cond_1d.py".

Then,

```python
$ python run/test_classifier_free_cond_1d.py
```

