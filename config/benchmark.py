import logging
from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory

from config.algorithm import FedAvg
from .algorithm import *

logger = logging.getLogger(__name__)

class BenchMark:
    def __init__(self, name, global_args, train_args, algorithm):
        logger.info(f"Initializing Benchmark {name}")
        self.name = name
        self.global_args = global_args
        self.train_args = train_args
        self.algorithm = algorithm

    def get_args(self):
        return self.global_args, self.train_args, self.algorithm

class DataSpecificBenchmark(BenchMark):
    def __init__(self, dataset_index):
        global_args = {
            'client_num': 10,
            'model': 'MLP',
            'dataset': dataset_index,  # Index from 1 to 15
            'batch_size': 32,
            'class_num': 5,  # Assuming a common number of classes
            'data_dir': './dataset',
            'communication_round': 1,
            'non-iid': False,
            'alpha': 1,
        }
        train_args = {
            'optimizer': 'SGD',
            'device': 'cpu',
            'lr': 1e-2,
            'weight_decay': 1e-5,
            'num_steps': 1,
        }
        self.algorithm = FedAvg() 
        super().__init__(f'DataBenchmark-{dataset_index}', global_args, train_args, self.algorithm)

def get_benchmark_by_index(dataset_index: int) -> BenchMark:
    if 1 <= dataset_index <= 15:
        return DataSpecificBenchmark(dataset_index)
    else:
        logger.error(f"Dataset index {dataset_index} is out of the allowed range (1-15)")
        raise ValueError(f"Dataset index {dataset_index} is out of the allowed range (1-15)")
