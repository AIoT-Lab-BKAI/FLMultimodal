from .dataset import PTBXLReduceDataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import os
import ujson
import importlib
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TaskPipe(IDXTaskPipe):
    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        class_path = feddata['datasrc']['class_path']
        class_name = feddata['datasrc']['class_name']
        origin_class = getattr(importlib.import_module(class_path), class_name)
        origin_train_data = cls.args_to_dataset(origin_class, feddata['datasrc']['train_args'])
        origin_test_data = cls.args_to_dataset(origin_class, feddata['datasrc']['test_args'])
        test_data = cls.TaskDataset(origin_test_data, [_ for _ in range(len(origin_test_data))])
        train_datas = []
        valid_datas = []
        modalities_list = []
        for name in feddata['client_names']:
            train_data = feddata[name]['dtrain']
            valid_data = feddata[name]['dvalid']
            if cls._cross_validation:
                k = len(train_data)
                train_data.extend(valid_data)
                random.shuffle(train_data)
                all_data = train_data
                train_data = all_data[:k]
                valid_data = all_data[k:]
            if cls._train_on_all:
                train_data.extend(valid_data)
            train_datas.append(cls.TaskDataset(origin_train_data, train_data))
            valid_datas.append(cls.TaskDataset(origin_train_data, valid_data))
            modalities_list.append(feddata[name]['modalities'])
            # modalities_list.append(list(range(12)))
        return train_datas, valid_datas, test_data, feddata['client_names'], modalities_list

def save_task(generator):
    """
    Store the splited indices of the local data in the original dataset (source dataset) into the disk as .json file
    The input 'generator' must have attributes:
        :taskpath: string. the path of storing
        :train_data: the training dataset which is a dict {'x':..., 'y':...}
        :test_data: the testing dataset which is a dict {'x':..., 'y':...}
        :train_cidxs: a list of lists of integer. The splited indices in train_data of the training part of each local dataset
        :valid_cidxs: a list of lists of integer. The splited indices in train_data of the valiadtion part of each local dataset
        :client_names: a list of strings. The names of all the clients, which is used to index the clients' data in .json file
        :source_dict: a dict that contains parameters which is necessary to dynamically importing the original Dataset class and generating instances
                For example, for MNIST using this task pipe, the source_dict should be like:
                {'class_path': 'torchvision.datasets',
                    'class_name': 'MNIST',
                    'train_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])','train': 'True'},
                    'test_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])', 'train': 'False'}
                }
        :return:
    """
    feddata = {
        'store': 'IDX',
        'client_names': generator.cnames,
        'dtest': [i for i in range(len(generator.test_data))],
        'datasrc': generator.source_dict
    }
    for cid in range(len(generator.cnames)):
        if generator.specific_training_leads:
            feddata[generator.cnames[cid]] = {
                'modalities': generator.specific_training_leads[cid],
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
        else:
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
    with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
        ujson.dump(feddata, outf)
    return
    

def iid_partition(generator):
    print(generator)
    labels = np.unique(generator.train_data.y)
    local_datas = [[] for _ in range(generator.num_clients)]
    for label in labels:
        permutation = np.random.permutation(np.where(generator.train_data.y == label)[0])
        split = np.array_split(permutation, generator.num_clients)
        for i, idxs in enumerate(split):
            local_datas[i] += idxs.tolist()
    return local_datas

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, local_hld_rate=0.0, seed=0, percentages=None, missing=False, modal_equality=False, modal_missing_case3=False, modal_missing_case4=False):
        super(TaskGen, self).__init__(benchmark='ptbxl_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/PTBXL_REDUCE',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        if self.dist_id == 0:
            self.partition = iid_partition
        self.num_classes = 10
        self.save_task = save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'benchmark.ptbxl_classification.dataset',
            'class_name': 'PTBXLReduceDataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'True',
                'train':'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'True',
                'train': 'False'
            }
        }
        self.missing = missing
        self.modal_equality = modal_equality
        self.modal_missing_case3 = modal_missing_case3
        self.modal_missing_case4 = modal_missing_case4
        self.specific_training_leads = None
        if self.missing and self.num_clients == 20:
            if self.modal_equality:
                self.specific_training_leads = [
                    (4, 7, 8, 9, 10, 11),
                    (0, 2, 5, 7, 9, 11),
                    (1, 2, 3, 7, 9, 11),
                    (1, 3, 4, 6, 7, 9),
                    (0, 1, 4, 5, 10, 11),
                    (0, 1, 2, 3, 8, 9),
                    (0, 1, 3, 6, 7, 8),
                    (2, 3, 4, 5, 7, 11),
                    (0, 3, 4, 7, 10, 11),
                    (1, 3, 4, 5, 7, 10),
                    (0, 3, 4, 9, 10, 11),
                    (0, 2, 3, 4, 7, 8),
                    (1, 3, 5, 6, 7, 8),
                    (0, 1, 5, 7, 8, 10),
                    (0, 6, 7, 8, 9, 11),
                    (0, 4, 5, 6, 7, 8),
                    (0, 5, 6, 7, 8, 9),
                    (0, 1, 2, 3, 5, 9),
                    (3, 4, 5, 7, 8, 9),
                    (1, 5, 7, 8, 9, 11)
                ]
                self.taskname = self.taskname + '_missing_modal_equality'
                self.taskpath = os.path.join(self.task_rootpath, self.taskname)
            elif self.modal_missing_case3:
                self.specific_training_leads = [
                    (1, 11), 
                    (0, 4, 6, 7, 10), 
                    (3, 6), 
                    (1, 2, 3, 4), 
                    (5,), 
                    (0, 1, 3, 5, 6, 9, 10), 
                    (1, 2, 4, 6, 8), 
                    (0, 1, 6), (1, 3, 8), 
                    (1, 3, 9, 10), 
                    (0,), 
                    (0, 3, 7, 8), 
                    (1, 5, 9, 11), 
                    (0, 2, 4, 5, 6, 7, 8), 
                    (0, 1, 2, 5, 6, 8), 
                    (4,), 
                    (1, 5), 
                    (3, 4, 8, 9), 
                    (6, 9), 
                    (0, 1, 3, 7, 8, 9, 10, 11)
                ]
                self.taskname = self.taskname + '_missing_modal_case3'
                self.taskpath = os.path.join(self.task_rootpath, self.taskname)
            elif self.modal_missing_case4:
                self.specific_training_leads = [
                    (1, 2, 3, 8, 10, 11), 
                    (1, 2, 5, 6, 7, 8), 
                    (0, 3, 4, 5, 6, 7, 8, 9, 10), 
                    (2, 5, 7, 9, 10, 11), 
                    (0, 1, 3, 4, 5, 6, 7, 8, 10, 11), 
                    (1, 3, 7, 9, 10, 11), 
                    (0, 1, 2, 3, 4, 6, 8, 9, 10, 11), 
                    (1, 2, 3, 5, 6, 7, 8, 10, 11), 
                    (0, 1, 2, 3, 4, 5, 6, 7, 9, 10), 
                    (2, 3, 4, 5, 7, 8, 11), 
                    (0, 2, 3, 4, 7, 8, 9, 10, 11), 
                    (0, 1, 2, 3, 5, 7, 9, 10, 11), 
                    (1, 3, 4, 5, 9, 10, 11), 
                    (0, 2, 4, 5, 6, 7, 9), 
                    (0, 1, 2, 4, 5, 7, 8, 10, 11), 
                    (1, 2, 4, 5, 6, 9), 
                    (0, 1, 4, 7, 9, 10), 
                    (0, 1, 3, 5, 6, 8, 9, 10, 11), 
                    (0, 3, 5, 6, 9, 10), 
                    (1, 2, 3, 5, 6, 7, 9)
                ]
                self.taskname = self.taskname + '_missing_modal_case4'
                self.taskpath = os.path.join(self.task_rootpath, self.taskname)
            else:
                self.specific_training_leads = [
                    (4, 5, 8),
                    (4, 5),
                    (2, 3, 5, 9),
                    (1, 3, 7, 8, 11),
                    (5, 6, 8, 9),
                    (0, 2, 3, 5, 8, 9),
                    (0, 2, 3, 5),
                    (0, 1, 3, 5),
                    (0, 3, 5, 10, 11),
                    (1, 4, 6),
                    (8, 9, 11),
                    (0, 3, 5, 6, 7, 11),
                    (2, 3, 4, 5, 7),
                    (0, 4, 7, 8),
                    (0, 3, 4, 6, 7),
                    (1, 5, 6, 7, 8),
                    (0, 1, 3, 4, 10),
                    (2, 4, 5, 7, 9, 11),
                    (3, 4, 5, 8, 10, 11),
                    (0, 1, 3, 7, 9, 11)
                ]
                self.taskname = self.taskname + '_missing'
                self.taskpath = os.path.join(self.task_rootpath, self.taskname)

    def load_data(self):
        self.train_data = PTBXLReduceDataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=True,
            train=True
        )
        self.test_data = PTBXLReduceDataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=True,
            train=False
        )
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = DataLoader

    def train_one_step(self, model, data, leads):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        loss, outputs = model(tdata[0], tdata[-1], leads)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            loss, outputs = model(batch_data[0], batch_data[-1], leads)
            total_loss += loss.item() * len(batch_data[-1])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / len(dataset),
            'acc': accuracy
        }

    @torch.no_grad()
    def server_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        for test_combi_index in range(len(leads)):
            total_loss = 0.0
            labels = list()
            predicts = list()    
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                loss, outputs = model(batch_data[0], batch_data[-1], leads[test_combi_index])
                total_loss += loss.item() * len(batch_data[-1])
                predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            
            result['loss'+str(test_combi_index+1)] = total_loss / len(dataset)
            result['acc'+str(test_combi_index+1)] = accuracy
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        # import pdb;pdb.set_trace()
        return result


    @torch.no_grad()
    def full_modal_server_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            loss, outputs = model(batch_data[0], batch_data[-1], leads)
            total_loss += loss.item() * len(batch_data[-1])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / len(dataset),
            'acc': accuracy
        }

    @torch.no_grad()
    def independent_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        for test_combi_index in range(len(leads)):
            labels = list()
            predicts = list()
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                predict = model.predict(batch_data[0], batch_data[-1], leads[test_combi_index])
                predicts.extend(predict.argmax(dim=1).cpu().tolist())
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            result['acc'+str(test_combi_index+1)] = accuracy
        return result
        
    @torch.no_grad()
    def independent_test_detail(self, model, dataset, leads, batch_size=64, num_workers=0):
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=1, num_workers=num_workers)
        labels = list()
        
        fin_output = []
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            fin_output.append(model.predict_detail(batch_data[0], batch_data[-1], leads))
        
        return fin_output