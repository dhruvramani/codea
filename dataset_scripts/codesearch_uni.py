import os
import json
import pandas as pd
from pathlib import Path

import torch
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

import dataset_scripts.utils as utils
from dataset_scripts.codesearch_multi import URLS, download_dataset

# NOTE - IMP, shift to datasets - memory efficient af.

class CodeSearchNetUnimodalDataset(Dataset):
    def __init__(self, config, tokenizer, ttype='train', preprocess_code=False):
        assert ttype in ['train', 'test', 'val']
        self.ttype = ttype
        self.config = config
        self.tokenizer = tokenizer
        self.preprocess_code = preprocess_code

        self._setup()

    def _setup(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

class CodeSearchNetUnimodalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        assert config.prog_lang in URLS.keys()

        self.config = config
        self.tokenizers = utils.get_tokenizer(config.prog_lang)
        
        if not (os.path.exists(config.data_path) and os.listdir(config.data_path)):
            self.prepare_data()

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding) 

    def val_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding)

    def test_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding)

    def prepare_data(self):
        download_dataset(self.config)

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = CodeSearchNetUnimodalDataset(self.config, self.tokenizer, ttype='train')
            self.val_dataset = CodeSearchNetUnimodalDataset(self.config, self.tokenizer, ttype='val')

        if stage == 'test' or stage == None:
            self.test_dataset = CodeSearchNetUnimodalDataset(self.config, self.tokenizer, ttype='test')

if __name__ == '__main__':
    print("Testing codesearch_uni.py")
    config = utils.get_test_config(model='gpt2', dataset='codesearch')

    datamodule = CodeSearchNetUnimodalDataModule(config)
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=1)
    print(datamodule.tokenizer.decode[for i in next(iter(train_loader))['input_ids']])