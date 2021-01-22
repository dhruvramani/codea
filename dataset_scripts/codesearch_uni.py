import os
import gzip
import pickle

from itertools import cycle

import torch
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

import dataset_scripts.utils as utils
from dataset_scripts.codesearch_multi import URLS, download_dataset
from dataset_scripts.codesearch_multi import CodeSearchNetMultimodalDataset

FILES = {'python': 'python_dedupe_definitions_v2.pkl', }

class CodeSearchNetUnimodalDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.cache = None
        self.cache_len = 10000
        self.prev_cache_idx = -1
        
        self.config.cache_path = os.path.join(self.config.cache_path, 'unimodal/')

        self._setup()

    def _setup(self): 
        cache_contents = os.listdir(self.config.cache_path)
        
        if cache_contents != []:
            self.len = torch.load(os.path.join(self.config.cache_path, 'len'))
        else:
            print("D CS-Uni : Creating cache.")
            d_path = os.path.join(self.config.data_path, FILES[self.config.prog_lang])
            with open(d_path, 'rb') as f:
                data = pickle.load(f)
            self.len = len(data)

            torch.save(self.len, os.path.join(self.config.cache_path, 'len'))
            print("Total files : ", self.len // self.cache_len)
            j = 0
            for i in range(112, self.len // self.cache_len):
                cached_features_file = os.path.join(self.config.cache_path, f'{i}.pt')
                features = []
                for k in range(i * self.cache_len, (i+1) * self.cache_len):
                    func = self.process_data(data[k]['function'])
                    features.extend(func)
                torch.save(features, cached_features_file)
                print(i, " saved.")
                j = i + 1

            if j * self.cache_len < self.len:
                cached_features_file = os.path.join(self.config.cache_path, f'{j}.pt')
                features = []
                for k in range(j * self.cache_len, self.len):
                    func = self.process_data(data[k]['function'])
                    features.extend(func)
                torch.save(features, cached_features_file)
                print(j)

            del data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cache_idx = idx // self.cache_len
        if cache_idx != self.prev_cache_idx:
            cache_file = os.path.join(self.config.cache_path, f'{cache_idx}.pt')
            self.cache = torch.load(cache_file)
            self.prev_cache_idx = cache_idx

        content = self.cache[idx % self.cache_len]
        return content             

    def process_data(self, raw_data):
        func = utils.preprocess_code(self.config, raw_data, nlines=False)
        tokenized_texts = self.tokenizer(func, add_special_tokens=False, truncation=False, max_length=utils.MAX_LENS[self.config.model])
        tokenized_texts = utils.group_texts(tokenized_texts, block_size=utils.MAX_LENS[self.config.model])
        return [{k: t[i] for k, t in tokenized_texts.items()} for i in range(len(tokenized_texts['input_ids']))]

class CodeSearchNetUnimodalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        assert config.prog_lang in URLS.keys()

        self.config = config
        self.tokenizer = utils.get_tokenizer(config)
        
        # if not (os.path.exists(config.data_path) and os.listdir(config.data_path)):
        #     self.prepare_data()

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer)) 

    def val_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))

    def test_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))

    # def prepare_data(self):
    #     download_dataset(self.config)

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = CodeSearchNetUnimodalDataset(self.config, self.tokenizer)
            # NOTE - uses multimodal dataset - change it later on.
            self.val_dataset = CodeSearchNetMultimodalDataset(self.config, [self.tokenizer], code_only=True, ttype='val')

        if stage == 'test' or stage == None:
            self.test_dataset = CodeSearchNetMultimodalDataset(self.config, [self.tokenizer], code_only=True, ttype='test')

if __name__ == '__main__':
    print("Testing codesearch_uni.py")
    config = utils.get_test_config(model='gpt2', dataset='codesearch')

    datamodule = CodeSearchNetUnimodalDataModule(config)
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=5)
    print(next(iter(train_loader)))
    val_loader = datamodule.val_dataloader(batch_size=5)
    print(next(iter(train_loader)))
    print(next(iter(val_loader)))

    # print([datamodule.tokenizer.decode(i) for i in next(iter(train_loader))['input_ids']])