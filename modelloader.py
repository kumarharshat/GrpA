import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import baseline_models
import LieModels



class ModelLoader():
    '''
    Model loader class creates the model given the model config file
    (Loading model functionality needs to be added)
    '''
    def __init__(self, config, dataset):
        self.config = config
        num_classes = dataset.config['num_targets']
        self.model = None
        if self.config['modeltype'] == 'regular': # FCNN-1
            self.model = baseline_models.regular1(input_size=self.config['input'],num_classes = num_classes)
        if self.config['modeltype'] == 'regular2': # FCNN-2
            self.model = baseline_models.regular2(input_size=self.config['input'],num_classes = num_classes)
        if self.config['modeltype'] == 'lie1': #GrpA
            self.model = LieModels.LieModel_SO3(self.config, dataset, output = num_classes)
        if self.config['modeltype'] == 'lieRot': # GrpA-Rot
            self.model = LieModels.LieModel_SO2(self.config, dataset)

        if self.model is None:
            assert False, "Model type does not match with existing implemented models"

        self.config['number_params'] = sum(p.numel() for p in self.model.parameters())