import logging
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
from config.algorithm import Algorithm
from server.aggregation_alg.fedavg import fedavgAggregator
from client.clients import Client, BaseClient, SignClient
from client.trainer.fedproxTrainer import fedproxTrainer
from client.trainer.SignTrainer import SignTrainer
from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory
from dataset.DatasetSpliter import DatasetSpliter
import numpy as np
from chainfl.interact import chain_proxy

class Task:
    '''
    Workflow of a Task:
    0. Construct (Model, Dataset) --> Benchmark
    1. Construct (Server, Client) --> FL Algorithm
    3. Process the dataset
    '''
    def __init__(self, global_args: dict, train_args: dict, algorithm: Algorithm):
        self.global_args = global_args
        self.train_args = train_args
        self.algorithm = algorithm 
        
        # Get Dataset
        dataset_factory = DatasetFactory(global_args['data_dir'])
        self.train_dataset = dataset_factory.get_dataset(global_args['dataset'], transform=global_args.get('train_transform'))
        self.test_dataset = dataset_factory.get_dataset(global_args['dataset'], transform=global_args.get('test_transform'))
        
        # Get Model
        self.model = ModelFactory().get_model(global_args['model'], global_args['class_num'])
        
        # Setup FL algorithm components
        self.server = algorithm.get_server()
        self.server = self.server()
        self.trainer = algorithm.get_trainer()
        self.client = algorithm.get_client()
        
        # Initialize client list and pool
        self.client_list = chain_proxy.get_client_list()
        print("client_list")
        print(self.client_list)
        self.client_pool = []

    def __repr__(self) -> str:
        return f"<Task Model: {self.model}, Dataset: {self.train_dataset}, Algorithm: {self.algorithm}>"
    

    def _construct_dataloader(self):
        logger.info("Constructing dataloader with batch size %d, client_num: %d, non-iid: %s", self.global_args.get('batch_size')
                    , chain_proxy.get_client_num(), "True" if self.global_args['non-iid'] else "False")
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataloader_list = DatasetSpliter().random_split(dataset     = self.train_dataset,
                                                                   client_list = chain_proxy.get_client_list(),
                                                                   batch_size  = batch_size)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)
    

    def _register_client(self):
    # Correcting method name from client_register to client_regist
        for client_id in self.client_list:
            chain_proxy.client_regist()  # Assuming you might need to do something with client_id here
        

    def _construct_client(self):
        print(self.client_list)
        for client_id in self.client_list:
            new_client = self.client(client_id, self.train_dataloader_list[client_id], self.model, self.trainer, self.train_args, self.test_dataloader, None)
            self.client_pool.append(new_client)

    def run(self):
        self._register_client()
        logger.info("check 1")
        self._construct_dataloader()
        logger.info("check 2")
        self._construct_client()
        logger.info("check 3")
        #self._construct_sign()
    
        for i in range(self.global_args['communication_round']):
            for client in self.client_pool:
                client.train(epoch=i)
                client.test(epoch=i)
                # client.sign_test(epoch=i)  # Include if sign testing is needed
            print("client_pool")
            print(self.client_pool)
            self.server.receive_upload(self.client_pool)
            global_model = self.server.aggregate()
            for client in self.client_pool:
                client.load_state_dict(global_model)

