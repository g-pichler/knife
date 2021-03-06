# Variational Information Bottleneck for Effective Low-resource Finetuning 

## Python requirements
This code is tested on:
- Python 3.7.7 
- transformers  2.8.0
- pytorch 1.2.0


## Donwloading the datasets
You can download the datasets from the following paths, which are from previously published papers, and put them all in "data/datasets/"
  - download the GLUE data from https://github.com/nyu-mll/GLUE-baselines, this include SNLI, MNLI, RTE, MRPC, STS-B

## Parameters in the code 
* ```num_samples``` Specifies the number of samples in case of running models on the subsampled datasets
* ```ib_dim``` Specifies the bottleneck size
* ```ib``` If this option is set, runs the VIBERT model
* ```deteministic``` If this option is set, runs the VIBERT model with beta=0
* ```beta``` Specifies the weight for the compression loss
* ```mixout``` defines the mixout propability
* ```weight_decay``` defines the weight for weight_decay regularization
* to run the model on the subsampled datasets, add ```--sample_train``` option and specify the number of 
samples with ```--num_samples N ```, where N is the number of samples.

## Usage
We provide the following sample scripts.
the bert model.
```
sh train.sh
```
