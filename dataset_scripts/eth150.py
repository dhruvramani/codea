import os
import glob
import pickle
import shutil
import urllib.request
from pathlib import Path
from itertools import cycle

import torch
import transformers
import pytorch_lightning as pl

from torch.utils.data import DataLoader, IterableDataset
from transformers import DataCollatorWithPadding

import dataset_scripts.utils as utils

URLS = {'javascript' : '',
        'python' : 'http://files.srl.inf.ethz.ch/data/py150_files.tar.gz'}

class ETH150Dataset(IterableDataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self._setup()

    def _setup(self):
        file_cache = os.path.join(self.config.cache_path, 'files.pkl')
        if not os.path.isfile(file_cache):
            data_dir = os.path.join(self.config.data_path, 'data/')
            self.files = Path(data_dir).rglob('*.py')
            self.files = list(map(lambda f: str(f.resolve()), self.files))
            
            with open(file_cache, 'wb') as f:
                pickle.dump(self.files, f)
        else:
            print("D ETH150 : Using cached file.")
            with open(file_cache, 'rb') as f:
                self.files = pickle.load(f)

    def stream(self, by_line=False):
        for cfile in self.files:
            if not(os.path.exists(cfile) and os.path.isfile(cfile)):
                continue
            with open(cfile,'r') as f:
                code = f.read()

            code = utils.preprocess_code(self.config, code, nlines=by_line)
            
            if by_line:
                for line in code.splitlines():
                    if line.strip() != '':
                        line = self.tokenizer(line, add_special_tokens=False)
                        yield line
            else:
                tokenized_texts = self.tokenizer(code, add_special_tokens=False, truncation=False, max_length=utils.MAX_LENS[self.config.model])
                tokenized_texts = utils.group_texts(tokenized_texts, block_size=utils.MAX_LENS[self.config.model])
                for i in range(len(tokenized_texts['input_ids'])):
                    yield {k: t[i] for k, t in tokenized_texts.items()}

    def __iter__(self):
        return cycle(self.stream())

class ETH150DataModule(pl.LightningDataModule):
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
        # TODO or default_data_collator (others below too)

    # def prepare_data(self):
    #     download_dataset(self.config)

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = ETH150Dataset(self.config, self.tokenizer)

def download_dataset(config):
    print("=> Downloading ETH150K {} dataset".format(config.prog_lang))

    file_path = os.path.join(config.data_path, URLS[config.prog_lang].split("/")[-1])
    urllib.request.urlretrieve(URLS[config.prog_lang], file_path)
    shutil.unpack_archive(file_path, config.data_path)

    file_path = os.path.join(config.data_path, 'data.tar.gz')
    shutil.unpack_archive(file_path, file_path.replace('.tar.gz', '/'))

    print("Downloading and processing completed.")

if __name__ == '__main__':
    print("Testing eth150.py")
    config = utils.get_test_config(model='gpt2', dataset='eth150')

    datamodule = ETH150DataModule(config)
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=5)
    #print(next(iter(train_loader)))
    # print([datamodule.tokenizer.decode(i) for i in next(iter(train_loader))['input_ids']])
    for i, sample in enumerate(train_loader):
        if i == 10:
            break
        op = [datamodule.tokenizer.decode([i]) for i in sample['input_ids'][0]]
        print(op)