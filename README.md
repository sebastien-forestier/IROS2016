# IROS2016
Repository aiming at reproducibility of experiments and analysis of results provided in IROS2016 conference paper. 

We provide source code of the experiments (Licence GPLv3), and data analysis. 

We do not provide data as it is 400GB

We explain how to re-generate some of it (generating all the exploration databases is 14 conditions x 100 trials x 5h).
## Paper
Here is the IROS [paper](http://sforestier.com/sites/default/files/Forestier2016Modular.pdf).
## Video 
Here is a [video](https://www.youtube.com/watch?v=NXXlPAycucY) of the setup. 

## Tutorial on Active Model Babbling
Here is a Jupyter Notebook explaining the Active Model Babbling algorithm with comparisons to other algorithms: [notebook](http://nbviewer.jupyter.org/github/sebastien-forestier/ExplorationAlgorithms/blob/master/main.ipynb).

## Experiments ##
* [notebook](http://nbviewer.jupyter.org/github/sebastien-forestier/IROS2016/blob/master/notebook/experiments.ipynb) describing how to run the experiments

## Analysis ##
* [notebook](http://nbviewer.jupyter.org/github/sebastien-forestier/IROS2016/blob/master/notebook/analysis.ipynb) describing how to analyze the data.

## Code Dependencies ##
* [Explauto](https://github.com/flowersteam/explauto) on branch random_goal_babbling, [this](https://github.com/flowersteam/explauto/commit/bda9e53b35aa036b2667945226ebca94fe89375c) commit
* [pydmps](https://github.com/sebastien-forestier/pydmps) on branch master, [this](https://github.com/sebastien-forestier/pydmps/commit/464450d99ec8be962d54270164861a56eb94993c) commit
* Run with Numpy version '1.10.1', and scipy version '0.15.1'.
