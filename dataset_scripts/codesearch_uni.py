import os
import pickle
from itertools import cycle

import torch
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, IterableDataset
from transformers import DataCollatorWithPadding

import dataset_scripts.utils as utils
from dataset_scripts.codesearch_multi import URLS, download_dataset
from dataset_scripts.codesearch_multi import CodeSearchNetMultimodalDataset

FILES = {'python': 'python_dedupe_definitions_v2.pkl', }

class CodeSearchNetUnimodalDataset(IterableDataset):
    def __init__(self, config, tokenizer):
        '''
        Unimodal code data, split by lines - for whole functions, use multimodal.
        # NOTE - Maybe, shift to datasets 
        # TODO - Cache the data 
        '''
        self.config = config
        self.tokenizer = tokenizer

        self._setup()

    def _setup(self):
        f_path = os.path.join(self.config.data_path, FILES[self.config.prog_lang])
        with open(f_path, 'rb') as f:
            self.data = pickle.load(f)
            self.dict_len = len(self.data)

    def stream(self, by_line=False):
        for dict_idx in range(self.dict_len):
            func = self.data[dict_idx]['function']
            func = utils.preprocess_code(self.config, func, nlines=by_line)
            
            if by_line:
                for line in func.splitlines():
                    if line.strip() != '':
                        line = self.tokenizer(line, add_special_tokens=False)
                        yield line
            else:
                tokenized_texts = self.tokenizer(func, add_special_tokens=False)
                tokenized_texts = utils.group_texts(tokenized_texts, block_size=self.tokenizer.model_max_length)
                for i in range(len(tokenized_texts['input_ids'])):
                    yield {k: t[i] for k, t in tokenized_texts.items()}

    def __iter__(self):
        return cycle(self.stream())

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
    # print([datamodule.tokenizer.decode(i) for i in next(iter(train_loader))['input_ids']])