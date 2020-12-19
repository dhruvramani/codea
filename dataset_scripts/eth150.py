import os
import shutil
import tarfile
import urllib.request

import torch
import datasets
import transformers
import pytorch_lightning as pl

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

import dataset_scripts.utils as utils

URLS = {'javascript' : '',
        'python' : 'http://files.srl.inf.ethz.ch/data/py150_files.tar.gz'}

FILES = {'python' : {'train' : 'python100k_train.txt', 
                     'test': 'python50k_eval.txt', 
                     'data': 'data/'},
         'javascript': {}}

class ETH150Dataset(Dataset):
    def __init__(self, config, tokenizer, all_train=True, ttype='train', preprocess_code=True):
        assert ttype in ['train', 'test']
        self.ttype = ttype
        self.config = config
        self.tokenizer = tokenizer
        self.preprocess_code = preprocess_code

        self._setup()

    def _setup(self):
        if self.ttype == 'train' and all_train:
            self.files = os.listdir(os.path.join(self.config.data_path, FILES[config.prog_lang]['data']))
        else':
            with open(os.path.join(self.config.data_path, FILES[config.prog_lang][self.ttype]), 'r') as f:
                self.files = f.readlines()

        self.dataset = load_dataset('text', data_files={self.ttype : self.files})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = self.dataset[idx]['text']
        if preprocess_code:
            utils.preprocess_code(line)
        return self.tokenizer(line, add_special_tokens=False)

class ETH150DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        assert config.prog_lang in URLS.keys()

        self.config = config
        self.tokenizer = utils.get_tokenizer(config.prog_lang)
        
        if not (os.path.exists(config.data_path) and os.listdir(config.data_path)):
            self.prepare_data()

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding) 
        # TODO or default_data_collator (others below too)

    def test_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding)

    def prepare_data(self):
        download_dataset(self.config)

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = ETH150Dataset(self.config, self.tokenizer, ttype='train')

        if stage == 'test' or stage == None:
            self.test_dataset = ETH150Dataset(self.config, self.tokenizer, ttype='test')

def download_dataset(config):
    print("=> Downloading ETH150K {} dataset".format(config.prog_lang))
    urllib.request.urlretrieve(URLS[config.prog_lang], config.data_path)
    
    file_path = os.path.join(config.data_path, URLS[config.prog_lang].split("/")[-1])
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall()
    tar.close()

    file_path = file_path.replace('.tar.gz', '/')
    files = os.listdir(source_dir)
    for file in files:
        shutil.move(os.path.join(file_path, file), config.data_path)

    shutil.rmtree(file_path)

    file_path = os.path.join(config.data_path, 'data.tar.gz')
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall()
    tar.close()
    print("Downloading and processing completed.")

if __name__ == '__main__':
    print("Testing eth150.py")
    config = utils.get_test_config(model='gpt2', dataset='eth150')

    datamodule = ETH150DataModule(config)
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=1)
    print(datamodule.tokenizer.decode[for i in next(iter(train_loader))['input_ids']])