# Domain adapation experiments


## Getting started

#### Requirements

This code was developped with:
    -python 3.8
    -torch 1.7

For the exhaustive list of packages, see `requirements.txt`.

#### Data

For MNIST and MNIST-M datasets, please download the archive file at [this link](https://drive.google.com/file/d/1SVcQTOkSb8vlc9vQO69UmfJdBZXXIcdz/view?usp=sharing) and untar this file:

```bash
tar -xvf data.tar.gz
```
The other datasets used will be automatically downloaded the firt time you launch the command.

## Tuning/Training the methods

We provide ready-to-use scripts that you can launch to train the methods. The best parameters for each methods are already set in each method-specific .yaml config file, available at config/. To launch a specific training, use:

```bash
bash scripts/train.sh <method> <source_data> <target_data> <seed>
```
For instance, to train KNIFE on the task MNIST -> MNIST-M on seed 0, use:
```bash
bash scripts/train.sh knife MNIST MNISTM 0
```

If you want to train a method on all tasks on 3 different seeds, as in the paper, please use:
```bash
bash scripts/train_all.sh <method>
```


## Inspecting the results

Once methods have been trained, we offer two ways to inspect results:

1) Plot several metrics, including validation and test accuracy, as well as mutual information:

```python
    python -m plot --folder <folder>
```
The command above will plot recursively all metrics of **all** experiments included in <folder>. For instance, if you use `python -m plot --folder results/` will plot all experients in folder, which will put on the same plot the metrics from possibly different tasks and methods. If you need a method or task specific, use this specific folder in the command.

2) Inspect per-dataset test performances of each method with:

```python
    python -m summarize_stats --group_by 'method' --folder 'results'
```
Note that test performance is taken as the test accuracy at the epoch corresponding to the best **validation** accuracy.