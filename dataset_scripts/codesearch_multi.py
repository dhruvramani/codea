import os
import json
import shutil
import zipfile
import pandas as pd
import urllib.request
from pathlib import Path

import torch
import datasets
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForTokenClassification

import dataset_scripts.utils as utils #dataset_scripts

# TODO - Add option for code2nl or nl2code

URLS = {'javascript' : 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip',
        'python' : 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip',
        'java' : 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip',
        'go' : 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip',}

# SOURCE : https://github.com/github/CodeSearchNet/blob/master/notebooks/ExploreData.ipynb
LEN = {'test': {'go': 14291, 'java': 26909, 'javascript': 6483, 'php' : 28391, 'python' : 22176, 'ruby' : 2279},
       'train': {'go': 317832, 'java': 454451, 'javascript': 123889, 'php': 523712, 'python': 412178, 'ruby': 48791},
       'valid': {'go': 14242, 'java': 15328, 'javascript': 8253, 'php': 26015, 'python': 23107, 'ruby': 2209}}

class CodeSearchNetMultimodalDataset(Dataset):
    def __init__(self, config, tokenizers, ttype='train', code_only=False):
        assert ttype in ['train', 'test', 'val']
        self.ttype = ttype
        self.config = config
        self.code_tokenizer = tokenizers[0]
        self.eng_tokenizer = tokenizers[-1] # NOTE : if common tokenizer, pass (tokenizer)
        self.code_only = code_only

        self.files = sorted(Path(self.config.data_path).glob('**/*{}*.gz'.format(self.ttype)))
        self.file_idx = 0
        self.prev_lens = 0

        self._setup()

    def _setup(self):
        columns = ['code', 'docstring']
        self.data = pd.read_json(self.files[self.file_idx], orient='records', compression='gzip', lines=True)[columns] 
    
    def __len__(self):
        return LEN[self.ttype][self.config.prog_lang]

    def __getitem__(self, idx):
        idx = idx - self.prev_lens
        if idx >= len(self.data.index):
            self.prev_lens += len(self.data.index)
            self.file_idx += 1
            self.file_idx = self.file_idx % len(self.files)
            self._setup()

        row = self.data.iloc[idx]
        code = row['code']
        
        code = utils.preprocess_code(self.config, code, nlines=False)
        code = self.code_tokenizer(code, add_special_tokens=False, truncation=False, max_length=utils.MAX_LENS[self.config.model])
        
        if not self.code_only:
            docstring = row['docstring']
            docstring = self.eng_tokenizer(docstring, add_special_tokens=False)
            code['labels'] = docstring['input_ids']
        
        return code

class CodeSearchNetMultimodalDataModule(pl.LightningDataModule):
    def __init__(self, config, common_tokenizer=True):
        super().__init__()
        assert config.prog_lang in URLS.keys()

        self.config = config
        self.tokenizers = [utils.get_tokenizer(config)]
        if not common_tokenizer:
            self.tokenizers[1] = utils.get_tokenizer(config.model) # DEBUG
        self.tokenizer = self.tokenizers[0]
        
        # if not (os.path.exists(config.data_path) and os.listdir(config.data_path)):
        #     self.prepare_data()

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=DataCollatorForTokenClassification(self.tokenizer)) 

    def val_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, collate_fn=DataCollatorForTokenClassification(self.tokenizer))

    def test_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, collate_fn=DataCollatorForTokenClassification(self.tokenizer))

    # def prepare_data(self):
    #     download_dataset(self.config)

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = CodeSearchNetMultimodalDataset(self.config, self.tokenizers, ttype='train')
            self.val_dataset = CodeSearchNetMultimodalDataset(self.config, self.tokenizers, ttype='val')

        if stage == 'test' or stage == None:
            self.test_dataset = CodeSearchNetMultimodalDataset(self.config, self.tokenizers, ttype='test')

def download_dataset(config):
    print("=> Downloading CodeSearchNet {} dataset".format(config.prog_lang))
    zip_path = os.path.join(config.data_path, '{}.zip'.format(config.prog_lang))
    urllib.request.urlretrieve(URLS[config.prog_lang], zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(config.data_path)

    source_dir = os.path.join(config.data_path, '{}/final/jsonl'.format(config.prog_lang))
    files = os.listdir(source_dir)
    for file in files:
        shutil.move(os.path.join(source_dir, file), config.data_path)

    shutil.rmtree(os.path.join(config.data_path, config.prog_lang))
    print("Downloading and processing completed.")

if __name__ == '__main__':
    print("Testing codesearch_multi.py")
    config = utils.get_test_config(model='p_codebert', dataset='codesearch')

    datamodule = CodeSearchNetMultimodalDataModule(config)
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=5)
    print(next(iter(train_loader)))
    # print([datamodule.tokenizer.decode(i) for i in next(iter(train_loader))['input_ids']])
    # print([datamodule.tokenizer.decode(i) for i in next(iter(train_loader))['labels']])