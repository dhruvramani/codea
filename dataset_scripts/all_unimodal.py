import os
import copy
from itertools import cycle

import torch
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, IterableDataset
from transformers import DataCollatorWithPadding

import utils as utils #dataset_scripts.utils
from bigcode import BigCodeDataset
from codesearch_uni import CodeSearchNetUnimodalDataset
from eth150 import ETH150Dataset

DATASETS = {'bigcode': BigCodeDataset, 'codesearch': CodeSearchNetUnimodalDataset, 'eth150': ETH150Dataset}

class AllUnimodalDataset(IterableDataset):
    def __init__(self, config, datasets, tokenizer, preprocess_code=True):
        self.config = config
        self.tokenizer = tokenizer

        self._setup(datasets, preprocess_code)

    def _setup(self, datasets, preprocess_code):
        self.datasets = []

        for ds_name in datasets:
            config = copy.deepcopy(self.config)
            config.dataset = ds_name
            config.data_path = os.path.join(config.data_path, '{}/{}/'.format(config.prog_lang, config.dataset))
            config.cache_path = os.path.join(config.cache_path, '{}/{}/'.format(config.prog_lang, config.dataset))
            
            ds = DATASETS[ds_name](config, self.tokenizer, preprocess_code=preprocess_code)
            ds = iter(ds)
            self.datasets.append(ds)

    def stream(self):
        counter = 0
        while True: 
            dataset = self.datasets[counter % len(self.datasets)]
            data = next(dataset)
            counter += 1
            yield data
    
    def __iter__(self):
        return cycle(self.stream())

class AllUnimodalDataModule(pl.LightningDataModule):
    def __init__(self, config, datasets=['bigcode', 'eth150']):
        super().__init__()
        assert set(datasets).issubset(DATASETS.keys())

        self.config = config
        self.datasets = datasets
        self.tokenizer = utils.get_tokenizer(config)

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer)) 
        # TODO or default_data_collator (others below too)

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = AllUnimodalDataset(self.config, self.datasets, self.tokenizer)

if __name__ == '__main__':
    print("Testing all_unimodal.py")
    config = utils.get_test_config(model='gpt2', dataset='all')

    datamodule = AllUnimodalDataModule(config, datasets=['bigcode', 'eth150'])
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=10)
    print(next(iter(train_loader)))
    # print([datamodule.tokenizer.decode(i) for i in next(iter(train_loader))['input_ids']])