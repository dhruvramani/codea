from random import randrange

import torch
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

import dataset_scripts.utils as utils
from dataset_scripts.bigcode import BigCodeDataset
from dataset_scripts.codesearch_uni import CodeSearchNetUnimodalDataset
from  dataset_scripts.eth150 import ETH150Dataset

DATASETS = {'bigcode': BigCodeDataset, 'codesearch': CodeSearchNetUnimodalDataset, 'eth150': ETH150Dataset}

class AllUnimodalDataset(Dataset):
    def __init__(self, config, datasets, tokenizer, preprocess_code=False):
        self.config = config
        self.datasets = [DATASETS[i](config, tokenizer, ttype='train', preprocess_code=preprocess_code) \
                        for i in datasets]
    
    def __len__(self):
        return sum([len(i) for i in self.datasets])

    def __getitem__(self, idx):
        dataset = self.datasets[idx % len(self.datasets)]
        data = dataset[idx % len(dataset)]
        return data

class AllUnimodalDataModule(pl.LightningDataModule):
    def __init__(self, config, datasets=['bigcode', 'codesearch']):
        super().__init__()
        assert set(datasets).issubset(DATASETS.keys())

        self.config = config
        self.datasets = datasets
        self.tokenizers = utils.get_tokenizer(config.prog_lang)

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding) 

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = AllUnimodalDataset(self.config, self.datasets, self.tokenizer)

if __name__ == '__main__':
    print("Testing all_unimodal.py")
    config = utils.get_test_config(model='gpt2', dataset='')

    datamodule = AllUnimodalDataModule(config, datasets=['bigcode', 'codesearch'])
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=4)
    print(datamodule.tokenizer.decode[for i in next(iter(train_loader))['input_ids']])