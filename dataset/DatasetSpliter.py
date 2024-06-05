import logging
import numpy
import numpy as np
import numpy.random
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict

logger = logging.getLogger(__name__)


class DatasetSpliter:
    '''
    Receive a dataset object. Provided with some method to random divided the dataset.
    
    For Federated Learning: 
    1. Random Split
    2. Non-IID Split with params of dirichlet distribution. 
    '''
    def __init__(self) -> None:
        return
    def _sample_random(self, dataset: Dataset, client_list: dict):
        
        return
    
    def _sample_dirichlet(self, dataset: Dataset, client_list: dict, alpha: float) -> defaultdict(list):
        if not client_list:
            logger.error("Client list is empty. Cannot proceed with data splitting.")
            raise ValueError("Client list cannot be empty before data splitting.")

        if not dataset:
            logger.error("Dataset is empty.")
            raise ValueError("Dataset cannot be empty.")
        if not client_list:
            logger.error("Client list is empty.")
            raise ValueError("Client list cannot be empty.")

        client_num = len(client_list)
        print(client_num)
        per_class_list = defaultdict(list)

        # Collect indices for each class
        for idx, (_, label) in enumerate(dataset):
            per_class_list[label].append(idx)

        # Distribute indices using Dirichlet distribution
        per_client_list = defaultdict(list)
        for label, indices in per_class_list.items():
            random.shuffle(indices)
            class_size = len(indices)
            proportions = numpy.random.dirichlet(np.ones(client_num) * alpha) * class_size
            start = 0
            for client_idx, proportion in enumerate(proportions):
                end = start + int(round(proportion))
                per_client_list[list(client_list.keys())[client_idx]].extend(indices[start:end])
                start = end

        logger.info("Dataset split among clients successfully.")
        return per_client_list

        
    def dirichlet_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32, alpha: int = 1) -> dict[DataLoader]:
        #get each client samples
        split_list = self._sample_dirichlet(dataset = dataset, 
                                            client_list = client_list,
                                            alpha = alpha)
        dataloaders = defaultdict(DataLoader)
        
        #construct dataloader
        for ind, (client_id, _) in enumerate(client_list.items()):
            indices = split_list[client_id]
            this_dataloader = DataLoader(dataset    = dataset,
                                         batch_size = batch_size,
                                         sampler    = SubsetRandomSampler(indices),
                                         num_workers= 4)
            dataloaders[client_id] = this_dataloader 
        
        return dataloaders
    
    def random_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32) -> dict[DataLoader]:
        #Here we use a large alpha to simulate the average sampling.
        return self.dirichlet_split(dataset, client_list, batch_size, 1000000)
    
    

    