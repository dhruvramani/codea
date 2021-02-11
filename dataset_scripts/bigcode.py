import os
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

# NOTE - IMP, old data, ~2015. Not to be used solely.

URLS = {'python' : {'data' : ('python-corpus.tar.gz', 'https://zenodo.org/record/3628784/files/python-corpus.tar.gz?download=1'),
                    'stats': ('python_dataset_stats.tar.gz', 'https://zenodo.org/record/3628784/files/python_dataset_stats.tar.gz?download=1'),},}

FILES = {'python' : {'train': 'python_dataset_stats/large_training_set.txt',
                     'val'  : 'python_dataset_stats/validation_set.txt',
                     'test' : 'python_dataset_stats/test_set.txt',
                     'dir'  : 'python-corpus/cleaned/'},}

class BigCodeDataset(IterableDataset):
    def __init__(self, config, tokenizer, ttype='train'):
        assert ttype in ['train', 'test', 'val']
        self.ttype = ttype
        self.config = config
        self.tokenizer = tokenizer

        self._setup()

    def _setup(self):
        lang_dat = FILES[self.config.prog_lang]
        dir_cache = os.path.join(self.config.cache_path, '{}_dir.pkl'.format(self.ttype))

        if not os.path.isfile(dir_cache):
            ref_file = lang_dat[self.ttype]
            ref_file = os.path.join(self.config.data_path, ref_file)

            self.dirs = []
            with open(ref_file, 'r') as rf:
                content = rf.read()
            for row in content.splitlines()[1:]:
                dire = row.split(',')[0]
                dire = os.path.join(self.config.data_path, lang_dat['dir'], dire)
                if os.path.isdir(dire):
                    self.dirs.append(dire)
            with open(dir_cache, 'wb') as f:
                pickle.dump(self.dirs, f)
        else:
            print('D BIGCODE : Using cache.')
            with open(dir_cache, 'rb') as f:
                self.dirs = pickle.load(f)

    def stream(self, by_line=False):
        for dire in self.dirs:
            # NOTE - writing this coz of modified repo structure
            dir_struct = dire.split('cleaned/')
            dir_struct[1] = '_{}/{}'.format(dir_struct[1][0] , dir_struct[1])
            dire = 'cleaned/'.join(dir_struct)
            files = Path(dire).rglob('*.py')
            files = list(map(lambda f: str(f.resolve()), files))
            for cfile in files:
                if not(os.path.exists(cfile) and os.path.isfile(cfile)):
                    continue
                with open(cfile, 'r', encoding='latin-1') as f:
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

class BigCodeDataModule(pl.LightningDataModule):
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
            self.train_dataset = BigCodeDataset(self.config, self.tokenizer, ttype='train')
            self.val_dataset = BigCodeDataset(self.config, self.tokenizer, ttype='val')

        if stage == 'test' or stage == None:
            self.test_dataset = BigCodeDataset(self.config, self.tokenizer, ttype='test')

def download_dataset(config):
    print("=> Downloading Bigcode {} dataset".format(config.prog_lang))

    for f, url in URLS[config.prog_lang].values():
        f = os.path.join(config.data_path, f)
        urllib.request.urlretrieve(url, f)
    
        dir_p = f.replace('.tar.gz', '/')
        shutil.unpack_archive(f, dir_p)

        if 'corpus' in dir_p:
            utils.change_subdir_sys(os.path.join(dir_p, 'cleaned/'))

    print("=> Download completed.")

if __name__ == '__main__':
    print("Testing bigcode.py")
    config = utils.get_test_config(model='gpt2', dataset='bigcode')

    datamodule = BigCodeDataModule(config)
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader(batch_size=1)
    # print(next(iter(train_loader)))
    #print([datamodule.tokenizer.decode(i) for i in next(iter(train_loader))['input_ids']])
    for i, sample in enumerate(train_loader):
        if i == 2:
            break
        op = [datamodule.tokenizer.decode(i) for i in sample['input_ids']]
        print(len(sample['input_ids'][0]), op)