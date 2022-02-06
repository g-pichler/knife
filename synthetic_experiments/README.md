## Entropy Estimation

Entropy estimation code and comparison with [Schrau.](https://nic.schraudolph.org/bib2html/b2hd-Schraudolph04.html) can be found in `./estimate_entropy.py`.
Run the `./estimate_entropy.ipynb` Jupyter notebook to reproduce the results. Tensorboard graphs and `.json` files are placed in `./runs`. 

## Mutual Information Estimation

The three experiments in Figure 3 can be reproduced using `./estimate_mi.ipynb`.

### Gaussian Distribution

We provide toy simulations in `mi_estimation.py` to show the estimation performance of CLUB and other MI estimators. The code in this section is written with [Pytorch](https://pytorch.org/) (latest version). 

The implementation of our KNIFE estimator, along with other baselines ([NWJ](https://media.gradebuddy.com/documents/2949555/12a1c544-de73-4e01-9d24-2f7c347e9a20.pdf), [MINE](http://proceedings.mlr.press/v80/belghazi18a), [InfoNCE](https://arxiv.org/pdf/1807.03748.pdf), [VUB](https://arxiv.org/abs/1612.00410), [CLUB](https://paperswithcode.com/paper/club-a-contrastive-log-ratio-upper-bound-of), [DOE](https://arxiv.org/pdf/1811.04251.pdf)), is in `mi_estimators.py`. 

The code in `mi_estimation.py` can also be executed directly
to demonstrate the MI estimation performance of different MI estimators.
The resulting graphics are saved as pdfs in the working directory.

### Uniform Distribution
We use the code of [DOE](https://arxiv.org/pdf/1811.04251.pdf), which is slightly adapted and placed in `doe/`.

The script `./doe/main.py` can also be run directly with the appropriate arguments.
Results are written as tensorboard graphs and `.json` files to `./results`.

## Total Running Time
Each experiement can be conducted on a computer with GPUs and run in less than two hours. 

## Requierements:

pandas > 1.0.0


## TODO : 
- add the MI in the tensorboard 
- longer :) etc... no stepping 