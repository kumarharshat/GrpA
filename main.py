import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.optim as optim
from tqdm import tqdm
import solver
import yaml
from dataloader import DataLoader
from modelloader import ModelLoader
import random
import os.path
import pickle
import time


# torch.manual_seed(1234)

def run_simulation(tconfig, modelconfig = None, verbose = False, seed = None):
    '''
    :param tconfig: training hyperparameters from 'trainingconfig.yaml'
    :param modelconfig: model hyperparameters from model specificed by 'trainingconfig.yaml'
    :param verbose: Whether to print intermediate values or not
    :param seed: Specify random seed
    :return: History (dict) of training
    '''

    # Load the dataset config
    with open(tconfig['Training']['Dataset'], "r") as stream:
        try:
            datasetconfig = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Load the model config
    if modelconfig is None:
        with open(tconfig['Training']['Model'], "r") as stream:
            try:
                modelconfig = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    # Specify training seeds
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        random.seed(tconfig['Training']['Randomseed'])
        torch.manual_seed(tconfig['Training']['Randomseed'])
        np.random.seed(tconfig['Training']['Randomseed'])

    # Get the path of the dataset
    datapath = os.getcwd() +datasetconfig['fileloc']  +datasetconfig['filename'] + '.pickle'
    # If the pickle file has been created, load it
    if os.path.isfile(datapath):
        dataloader = pickle.load(open(datapath, "rb"))
    else:
        # Otherwise create the pickle file and save it
        dataloader = DataLoader(datasetconfig)
        pickle.dump(dataloader, open(datapath, "wb"))

    # Use small dataset for debugging
    if tconfig['Training']['debug']:
        debug_size = 256
        p = np.random.permutation(dataloader.dataset.test_loader.__len__())
        small = p[0:debug_size]
        dataloader.dataset.test_loader =torch.utils.data.Subset(dataloader.dataset.test_loader, small)
        p = np.random.permutation(dataloader.dataset.train_loader.__len__())
        small = p[0:debug_size]
        dataloader.dataset.train_loader =torch.utils.data.Subset(dataloader.dataset.train_loader, small)
        p = np.random.permutation(dataloader.dataset.val_loader.__len__())
        small = p[0:debug_size]
        dataloader.dataset.val_loader =torch.utils.data.Subset(dataloader.dataset.val_loader, small)

    # Load the model
    model = ModelLoader(modelconfig, dataset = dataloader)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model.model.to(device)

    # Initialize the solver
    main_solver = solver.Solver(dataloader, model, tconfig['Training'],verbose = verbose)

    # Get start time
    start = time.time()

    # Train the model
    main_solver.train()

    # Print the total time to train
    end = time.time()
    print(end - start)
    main_solver.save_history(end - start)

    # Return the solver history
    return main_solver.History



# # Comment this when runnign the tuning file
# with open("trainingconfig.yaml", "r") as stream:
#     try:
#         tconfig = yaml.safe_load(stream)
#         # print(yaml.safe_load(stream))
#     except yaml.YAMLError as exc:
#         print(exc)
#
# run_simulation(tconfig, verbose = True)