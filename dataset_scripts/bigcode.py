import os
import urllib.request

import torch
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

import dataset_scripts.utils as utils

# NOTE - IMP, shift to datasets - memory efficient af.

URLS = {'python' : {'train' : ('large_training_set_pre', 'https://zenodo.org/record/3628636/files/large_training_set_pre?download=1'),
                    'test'  : ('test_set_pre', 'https://zenodo.org/record/3628636/files/test_set_pre?download=1'),
                    'val'   : ('val_set_pre', 'https://zenodo.org/record/3628636/files/testProjects?download=1')},}

class BigCodeDataset(Dataset):
    def __init__(self, config, tokenizer, ttype='train', preprocess_code=True):
        assert ttype in ['train', 'test', 'val']
        
        self.config = config
        self.tokenizer = tokenizer
        self.preprocess_code = preprocess_code

        self.lang_dat = URLS[config.prog_lang]
        self.data_file = os.path.join(config.data_path, self.lang_dat[ttype][0])
        self.cache_path = os.path.join(config.cache_path, self.lang_dat[ttype][0])

        self.file_index, self.len = utils.index_file(self.data_file, self.cache_path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with open(self.data_file, 'rb', encoding='utf-8') as f:
            f.seek(self.file_index[idx])
            line = f.readline()

        if self.preprocess_code:
            line = utils.preprocess_code(line)
        
        return self.tokenizer(line, add_special_tokens=False)

class BigCodeDataModule(pl.LightningDataModule):
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
            self.train_dataset = BigCodeDataset(self.config, self.tokenizer, ttype='train')
            self.val_dataset = BigCodeDataset(self.config, self.tokenizer, ttype='val')

        if stage == 'test' or stage == None:
            self.test_dataset = BigCodeDataset(self.config, self.tokenizer, ttype='test')

def download_dataset(config):
    print("=> Downloading Bigcode {} dataset".format(config.prog_lang))

    for file, url in URLS[config.prog_lang].values():
        file = os.path.join(config.data_path, file)
        urllib.request.urlretrieve(url, file)

    print("=> Download completed.")

if __name__ == '__main__':
    print("Testing bigcode.py")
    config = utils.get_test_config(model='gpt2', dataset='bigcode')

    datamodule = BigCodeDataModule(config)
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=1)
    print(datamodule.tokenizer.decode[for i in next(iter(train_loader))['input_ids']])