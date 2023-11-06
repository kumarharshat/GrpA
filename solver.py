import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import time
import os.path
from sklearn.metrics import confusion_matrix

'''
Solver class takes dataset class, model class, and training config settings and runs the training loop 
'''
class Solver():
    def __init__(self, dataset, model, settings, verbose = False, **kwargs):
        self.num_targets = dataset.config['num_targets']
        self.model = model.model
        self.settings = settings
        self.iterations = settings['iterations']
        self.optimizer =optim.Adam(self.model.parameters(), lr=self.settings['learningrate'], weight_decay= self.settings['weightdecay'])
        self.trainset = DataLoader(dataset.dataset.train_loader, batch_size=self.settings['batchsize'], shuffle=True)
        self.valset = DataLoader(dataset.dataset.val_loader, batch_size=self.settings['batchsize'], shuffle=True)
        self.testset = DataLoader(dataset.dataset.test_loader, batch_size=self.settings['batchsize'], shuffle=True)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.criterion = torch.nn.CrossEntropyLoss()
        self.verbose = verbose
        self.History ={}
        self.History['training_loss'] = np.zeros(self.iterations)
        self.History['model_config'] = model.config
        self.History['dataset_config'] = dataset.config
        self.History['training_config'] = self.settings
        self.History['training_accuracy'] = []
        self.History['validation_accuracy'] = []
        self.History['test_accuracy'] = []
        self.History['training_accuracy_ep'] = []
        self.History['validation_accuracy_ep'] = []
        self.History['test_accuracy_ep'] = []
        self.History['test_epsilon'] = []
        self.History['validation_epsilon'] = []
        self.History['training_epsilon'] = []
        self.History['training_cm'] = []
        self.History['val_cm'] = []
        self.History['test_cm'] = []

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        if self.verbose:
            print([p.numel() for p in self.model.parameters()])
            print([p.numel() for p in self.model.parameters()][-2]/pytorch_total_params)
        self.History['num_params'] = pytorch_total_params
        if hasattr(dataset.dataset, 'loss_score'):
            self.History['loss_score'] = dataset.dataset.loss_score

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        # TODO: Add scheudler

    def eval_accuracy(self, dset, epsilon_eval = False):
        '''

        :param dset: which dataset are we evaluating
        :return: accuracy of self.model
        '''
        if dset =='train':
            dataset = self.trainset
        elif dset == 'val':
            dataset = self.valset
        elif dset =='test':
            dataset = self.testset
        num_correct = 0
        for batch_ in dataset:
            x = batch_[0]
            y = batch_[1]
            self.optimizer.zero_grad()
            if epsilon_eval:
                yhat = self.model.forward(x.to(self.device), add_noise = True).argmax(dim=1).cpu()
            else:
                yhat = self.model.forward(x.to(self.device)).argmax(dim =1).cpu()
            num_correct += torch.sum(yhat == y)
        if dset == 'train':
            if epsilon_eval:
                self.History['training_accuracy_ep'].append(num_correct / dataset.dataset.__len__())
            else:
                self.History['training_accuracy'].append(num_correct / dataset.dataset.__len__())
        elif dset == 'val':
            if epsilon_eval:
                self.History['validation_accuracy_ep'].append(num_correct / dataset.dataset.__len__())
            else:
                self.History['validation_accuracy'].append(num_correct / dataset.dataset.__len__())
        elif dset =='test':
            if epsilon_eval:
                self.History['test_accuracy_ep'].append(num_correct / dataset.dataset.__len__())
            else:
                self.History['test_accuracy'].append(num_correct / dataset.dataset.__len__())
        if self.verbose:
            print(dset, num_correct / dataset.dataset.__len__())
        return


    def eval_epsilon(self, dset):
        '''

        :param dset: which dataset are we evaluating
        :return: accuracy of self.model
        '''
        if dset =='train':
            dataset = self.trainset
        elif dset == 'val':
            dataset = self.valset
        elif dset =='test':
            dataset = self.testset
        norm_difference = 0
        for batch_ in dataset:
            x = batch_[0]
            y = batch_[1]
            self.optimizer.zero_grad()
            yhat_epsilon = self.model.forward(x.to(self.device), add_noise = True)
            yhat = self.model.forward(x.to(self.device), add_noise = False)
            norm_difference += torch.norm(yhat - yhat_epsilon).item()
        if dset == 'train':
            self.History['training_epsilon'].append(norm_difference / dataset.dataset.__len__())
        elif dset == 'val':
            self.History['validation_epsilon'].append(norm_difference / dataset.dataset.__len__())
        elif dset =='test':
            self.History['test_epsilon'].append(norm_difference / dataset.dataset.__len__())
        if self.verbose:
            print(dset, norm_difference / dataset.dataset.__len__())
        return


    def eval_cm(self, dset):
        '''

        :param dset: which dataset are we evaluating
        :return: accuracy of self.model
        '''
        if dset =='train':
            dataset = self.trainset
        elif dset == 'val':
            dataset = self.valset
        elif dset =='test':
            dataset = self.testset
        num_correct = 0
        yhat = []
        ytrue = []
        for batch_ in dataset:
            x = batch_[0]
            y = batch_[1]
            self.optimizer.zero_grad()
            yhat.append(self.model.forward(x.to(self.device)).argmax(dim =1).cpu())
            ytrue.append(y)

        yhat = np.concatenate(yhat)
        ytrue = np.concatenate(ytrue)
        cm = confusion_matrix(ytrue,yhat, labels = range( self.num_targets))
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        if dset == 'train':
            self.History['training_cm'].append(cm)
        elif dset == 'val':
            self.History['val_cm'].append(cm)
        elif dset =='test':
            self.History['test_cm'].append(cm)
        if self.verbose:
            print(dset, cm)
        return


    def save_history(self, time1 = None):
        if self.settings['Saving']['subfolder'] == 'dateTime':
            timestr = time.strftime("%Y%m%d")
        else:
            timestr =  self.settings['Saving']['subfolder']
        save_path = os.getcwd() + '/simulations/' + timestr + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        name_of_file = self.settings['Saving']['filename']

        completeName = os.path.join(save_path, name_of_file + ".pickle")
        if time1 is not None:
            self.History['time'] = time1
        pickle.dump([self.History], open(completeName, "wb"))

    def train(self):
        iter = 0
        while iter < self.iterations:
            if iter % 5 == 0 and self.verbose:
                print(iter)
            num_correct = 0
            for batch_ in self.trainset:
                x = batch_[0]
                y = batch_[1]
                self.optimizer.zero_grad()
                self.model.train()
                yhat = self.model.forward(x.to(self.device)).cpu()
                self.model.eval()
                loss = self.criterion(yhat, y)
                num_correct += torch.sum(yhat.argmax(dim =1).cpu() == y)
                self.History['training_loss'][iter] += loss.detach()
                loss.backward()
                self.optimizer.step()
            self.History['training_accuracy'].append(num_correct / self.trainset.dataset.__len__())
            if iter % self.settings['Saving']['accuracyfreq'] == 0:
                self.eval_accuracy('val', epsilon_eval=True)
                self.eval_accuracy('test', epsilon_eval=True)
                self.eval_accuracy('train', epsilon_eval=True)
                self.eval_accuracy('val')
                self.eval_accuracy('test')
                # self.eval_accuracy('train')
                self.eval_epsilon('val')
                self.eval_epsilon('test')
                self.eval_epsilon('train')

            iter += 1

        self.eval_cm('test')
        if self.verbose:
            print('Parameters', [p.numel() for p in self.model.parameters()])
            print('Total Parameters',np.sum([p.numel() for p in self.model.parameters()]))
        return

    def plot(self):
        return

    def save(self):
        return