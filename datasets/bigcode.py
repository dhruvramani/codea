import os
import sys
import shutil 
import tarfile
import urllib.request

import torch
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

URLS = {'python' : 'https://ndownloader.figshare.com/files/21493464',
        'c'      : 'https://ndownloader.figshare.com/files/21538806'}

class BigCodeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        assert config.prog_lang in URLS.keys()

        self.config = config
        self.tokenizer = NotImplemented
        
        if not (os.path.exists(config.data_path) and os.listdir(config.data_path)):
            self.prepare_data()

    def train_dataloader(self, batch_size=None): # NOTE fix batch_size ?
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding) # or default_data_collator?

    def prepare_data(self):
        download_dataset(self.config)

    def setup(self, stage=None):
        raise NotImplementedError
        # if stage == 'fit' or stage == None:
        #     self.train_dataset = # TODO (ttype='train')
        #     self.val_dataset = # TODO (ttype='val')

        # if stage == 'test' or stage == None:
        #     self.test_dataset = # TODO (ttype='test')

def download_dataset(config):
    c_path = os.path.join(config.data_path, '{}-corpus.tar.gz'.format(config.prog_lang))
    d_path = os.path.join(config.data_path, '{}-corpus/'.format(config.prog_lang))
    
    urllib.request.urlretrieve(URLS[config.prog_lang], c_path)
    tar = tarfile.open(c_path, 'r:gz')
    tar.extractall(path=config.data_path)
    tar.close()

    for file in os.listdir(d_path):
        shutil.move(os.path.join(d_path, file), os.path.join(config.data_path, file))

    os.rmdir(d_path)
    #os.remove(c_path)