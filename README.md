# GrpA
Code to implement the Lie Group Algebra convolutional filters as given by [our paper](https://arxiv.org/pdf/2210.17425.pdf).

## Installation

In your working folder, run 
```commandline
$ git clone FOLDER
```

Then, activate the conda enviromment by running 

```commandline
$ conda env create -f environment.yaml
$ conda activate grpa
```

We highly recommend using pytorch with CUDA enabled, as there are many computations which are very computationally expenseive on CPU only. However, if CUDA is not available, please do the following changes to the `environment.yaml` file: 

1) Remove `nvidia` from `channels`
2) Change `nvidia::cudatoolkit>10.0` to to `cpuonly`

In order to run the ModelPoint10 simulations, we need the additional commands after activating `grpa`
```commandline
$ pip install torch_geometric
$ pip install torch_points3d
```
Note that these take a while (around a half hour). Please also uncomment `pointcloud.py`
Torch version that works for me is `1.13.1`. 


# Dataset requirements
The rotated MNIST dataset will need to be downloaded and added to the '/data' folder. It can be obtained by the following commands
```
$ wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
$ unzip -n mnist_rotation_new.zip -d data
```

ModelPoint10 dataset will be downloaded automatically when run. Knot dataset will also be generated automatically when run.

# How to run the code
See the `trainingconfig.yaml` to choose the parameters with which to run the code.
Then you can run a single simulation with the command
```commandline
python single_simulation.py
```
Tune tasks can be run using the `modeltunetask.yaml` and `tunetask.yaml` config files with the `tunewrapper_models.py` and `tunewrapper.py` python scripts respectively.
