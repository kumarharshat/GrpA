# LieSP

## Installation

In your working folder, run 
```commandline
$ git clone FOLDER
```

Then, activate the conda enviromment by running 

```commandline
$ conda env create -f environment.yaml
$ conda activate liesp
$ pip install torch_geometric
$ pip install torch_points3d
```

We highly recommend using pytorch with CUDA enabled, as there are many computations which are very computationally expenseive on CPU only. However, if CUDA is not available, please do the following changes to the `environment.yaml` file: 

1) Remove `nvidia` from `channels`
2) Change `nvidia::cudatoolkit>10.0` to to `cpuonly`



# Dataset requirements
You will need to download the data if you want to change the dataset config files. This can be done in the following way:

# How to run the code



