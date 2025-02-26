from ...fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections
import utils.fflow as flw
import os
import torch
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 5
        self.list_testing_leads = [
            [1],                             #1
            [1, 3],                          #2
            [0, 1, 3],                   #3
            [0, 1, 2, 3],                #4
            [0, 1, 2, 3, 4],             #5
        ]
        self.lr = option['learning_rate']

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        flw.logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds+1):
            self.current_round = round
            ss.clock.step()
            # using logger to evaluate the model
            flw.logger.info("--------------Round {}--------------".format(round))
            flw.logger.time_start('Time Cost')
            if flw.logger.check_if_log(round, self.eval_interval) and round > 1:
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_once()
                flw.logger.time_end('Eval Time Cost')
            # check if early stopping
            if flw.logger.early_stop(): break
            # federated train
            self.iterate()
            # decay learning rate
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        # save results as .json file
        flw.logger.save_output_as_json()
        return

    def iterate(self):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        conmmunitcation_result = self.communicate(self.selected_clients)
        models = conmmunitcation_result['model']
        modalities_list = conmmunitcation_result['modalities']
        self.model = self.aggregate(models, modalities_list)
        return

    @torch.no_grad()
    def aggregate(self, models: list, modalities_list: list, beta1=0.9, beta2=0.999, epsilon=1e-8, m=None, v=None):
        beta1 = 0.8
        beta2 = 0.99
        epsilon = 1e-4
        print(beta1, beta2, epsilon)
        new_model = copy.deepcopy(self.model)
        t = self.current_round
        
        for lead in range(self.n_leads):
            global_dict = new_model.feature_extractors[lead].state_dict()
            if new_model.feature_extractors[lead].m is None:
                new_model.feature_extractors[lead].m, new_model.feature_extractors[lead].v = {}, {}
                for k in global_dict.keys():
                    new_model.feature_extractors[lead].m[k], new_model.feature_extractors[lead].v[k] = torch.zeros_like(global_dict[k]), torch.zeros_like(global_dict[k])

            for k in global_dict.keys():
                # updates = torch.stack([model.feature_extractors[lead].state_dict()[k] - global_dict[k] for model in models], 0).mean(0)
                updates = torch.stack([model.feature_extractors[lead].state_dict()[k].float() - global_dict[k].float() for model in models], 0).mean(0)
                new_model.feature_extractors[lead].m[k] = beta1 * new_model.feature_extractors[lead].m[k] + (1 - beta1) * updates
                new_model.feature_extractors[lead].v[k] = beta2 * new_model.feature_extractors[lead].v[k] + (1 - beta2) * updates.pow(2)

                m_hat = new_model.feature_extractors[lead].m[k] / (1 - beta1 ** t)
                v_hat = new_model.feature_extractors[lead].v[k] / (1 - beta2 ** t)

                # global_dict[k] += self.lr * m_hat / (v_hat.sqrt() + epsilon)
                update = self.lr * m_hat / (v_hat.sqrt() + epsilon)
                # print(update)
                # update_step = torch.clamp(update_step, -0.01, 0.01)
                global_dict[k] = global_dict[k].float() + self.lr * m_hat / (v_hat.sqrt() + epsilon)

            new_model.feature_extractors[lead].load_state_dict(global_dict)
        
        global_dict = new_model.classifier.state_dict()
        if new_model.classifier.m is None:
            new_model.classifier.m, new_model.classifier.v = {}, {}
            for k in global_dict.keys():
                new_model.classifier.m[k], new_model.classifier.v[k] = torch.zeros_like(global_dict[k]), torch.zeros_like(global_dict[k])

        for k in global_dict.keys():
            updates = torch.stack([model.classifier.state_dict()[k] - global_dict[k] for model in models], 0).mean(0)
            new_model.classifier.m[k] = beta1 * new_model.classifier.m[k] + (1 - beta1) * updates
            new_model.classifier.v[k] = beta2 * new_model.classifier.v[k] + (1 - beta2) * updates.pow(2)

            m_hat = new_model.classifier.m[k] / (1 - beta1 ** t)
            v_hat = new_model.classifier.v[k] / (1 - beta2 ** t)

            update = self.lr * m_hat / (v_hat.sqrt() + epsilon)
            # print(update)
            global_dict[k] += update

        new_model.classifier.load_state_dict(global_dict)

        return new_model
    
    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        # return dict()
        if model is None: model=self.model
        if self.test_data:
            return self.calculator.server_test(
                model=model,
                dataset=self.test_data,
                batch_size=self.option['test_batch_size'],
                leads=self.list_testing_leads
            )
        else:
            return None

    def test_on_clients(self, dataflag='train'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            dataflag: choose train data or valid data to evaluate
        :return
            metrics: a dict contains the lists of each metric_value of the clients
        """
        all_metrics = collections.defaultdict(list)
        for client_id in self.selected_clients:
            # import pdb; pdb.set_trace()
            # print(client_id)
            c = self.clients[client_id]
            client_metrics = c.test(self.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics


class Client(BasicClient):
    def __init__(self, option, modalities, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads = 5
        self.modalities = modalities

    def pack(self, model):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
            "modalities": self.modalities
        }

    @ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        model.train()
        optimizer = self.calculator.get_optimizer(
            model=model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            if batch_data[-1].shape[0] == 1:
                continue
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            # import pdb; pdb.set_trace()
            # print(iter, batch_data[-1].shape[0])
            loss, outputs = self.calculator.train_one_step(
                model=model,
                data=batch_data,
                leads=self.modalities
            )['loss']
            loss.backward()
            optimizer.step()
        return

    @fmodule.with_multi_gpus
    def test(self, model, dataflag='train'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if dataflag == "train":
            dataset = self.train_data
        elif dataflag == "valid":
            dataset = self.valid_data
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # print(len(dataset))
        return self.calculator.test(
            model=model,
            dataset=dataset,
            leads=self.modalities
        )