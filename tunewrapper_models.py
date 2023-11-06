import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.optim as optim
from tqdm import tqdm
import solver
import yaml
from dataloader import DataLoader
from modelloader import ModelLoader
from main import run_simulation
import itertools
import os.path
import time
import pickle
def create_tasks(tuning_params):
    param = []
    lists = []
    for el in tuning_params:
        param.append(el[0])
        lists.append(el[1])
    lists1 = list(itertools.product(*lists))
    return param, lists1


if __name__ == '__main__':
    with open("modeltunetask.yaml", "r") as stream:
        try:
            tune_config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    tuning_params = []
    for key_ in tune_config['Model Config Tune']:
        if 'config' not in key_ and key_ != 'Task Name':
            tuning_params.append((key_, tune_config['Model Config Tune'][key_]))

    params, lists = create_tasks(tuning_params)
    list_of_sims = []
    with open(tune_config['config'], "r") as stream:
        try:
            tconfig = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    filename_base = tconfig['Training']['Saving']['filename']
    for sim_num, vals_ in enumerate(lists):
        with open('model_configs/' + tune_config['modelconfig'], "r") as stream:
            try:
                modelconfig = yaml.safe_load(stream)
                # print(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
        sim_ = {}
        print(params, vals_)

        for param_, val_ in zip(params, vals_):
            if param_ in ['Randomseed']:
                tconfig[param_] = val_
            else:
                modelconfig[param_] = val_
                sim_[param_] = val_
        tconfig['Training']['Saving']['filename'] =filename_base+ '_' + str(sim_num)
        History = run_simulation(tconfig, modelconfig, seed = sim_num)
        print('Training Acc', History['training_accuracy'][-1], 'Val Acc', History['validation_accuracy'][-1], 'Test', History['test_accuracy'][-1])
        print('Training Acc', np.max(History['training_accuracy']), 'Val Acc', np.max(History['validation_accuracy']), 'Test', np.max(History['test_accuracy']))
        sim_['History'] = History
        list_of_sims.append(sim_)

    if tconfig['Training']['Saving']['subfolder'] == 'dateTime':
        timestr = time.strftime("%Y%m%d")
    else:
        timestr = tconfig['Training']['Saving']['subfolder']

    save_path = os.getcwd() + '/simulations/' + timestr + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    name_of_file = tune_config['logging']
    completeName = os.path.join(save_path, name_of_file + ".pickle")
    pickle.dump(list_of_sims, open(completeName, "wb"))
    print('Finished tune task')


