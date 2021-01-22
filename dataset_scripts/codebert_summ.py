import os
import sys
import gzip
import json
import numpy as np
import pandas as pd
from more_itertools import chunked

import torch
import pytorch_lightning as pl 

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

''' SOURCE https://github.com/microsoft/CodeBERT/blob/master/code2nl/run.py '''

class CodeBertSummDataset(Dataset):
    def __init__(self, config, tokenizer, ttype):
        self.config = config
        self.tokenizer = tokenizer

        self.files = {'train' : 'train.jsonl', 'dev' : 'valid.jsonl', 'test' : 'test.jsonl'}
        self.cache_dir = os.path.join(self.config.cache_path, 'cached_{}_{}_{}/'.format(ttype, self.files[ttype].split('.')[0], self.config.max_seq_length))
        
        self.cache_len = 5000
        self.prev_cache_idx = -1
        self.cache = None

        self._setup(ttype)

    def _setup(self, ttype='train'):
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
            cache_contents = []
        else:
            cache_contents = os.listdir(self.cache_dir)

        if cache_contents != []:
            self.len = torch.load(os.path.join(self.cache_dir, 'len'))
        else:
            print("D CS-B - Creating cache.")
            examples = read_examples(os.path.join(self.config.data_path, self.files[ttype]))

            features = self.convert_examples_to_features(examples, stage=ttype)
            self.len = len(features)

            torch.save(self.len, os.path.join(self.cache_dir, 'len'))

            j = 0
            for i in range(self.len // self.cache_len):
                cached_features_file = os.path.join(self.cache_dir, f'{i}.pt')
                torch.save(features[i * self.cache_len : (i + 1) * self.cache_len], cached_features_file)
                print(i, " saved")
                j = i + 1

            if j * self.cache_len < self.len:
                cached_features_file = os.path.join(self.cache_dir, f'{j}.pt')
                torch.save(features[j * self.cache_len :], cached_features_file)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cache_idx = idx // self.cache_len
        if cache_idx != self.prev_cache_idx:
            cache_file = os.path.join(self.cache_dir, f'{cache_idx}.pt')
            self.cache = torch.load(cache_file)
            self.prev_cache_idx = cache_idx

        content = self.cache[idx % self.cache_len]
        content = {'source_ids' : content.source_ids, 'target_ids': content.target_ids, 'source_mask': content.source_mask, 'target_mask': content.target_mask}
        return content  

    def convert_examples_to_features(self, examples, max_source_length=256, max_target_length=128, stage=None):
        features = []
        for example_index, example in enumerate(examples):
            source_tokens = self.tokenizer.tokenize(example.source)[:max_source_length - 2]
            source_tokens = [self.tokenizer.cls_token] + source_tokens + [self.tokenizer.sep_token]
            source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens) 
            source_mask = [1] * (len(source_tokens))
            padding_length = max_source_length - len(source_ids)
            source_ids += [self.tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length
     
            if stage=="test":
                target_tokens = self.tokenizer.tokenize("None")
            else:
                target_tokens = self.tokenizer.tokenize(example.target)[: max_target_length - 2]
            target_tokens = [self.tokenizer.cls_token] + target_tokens + [self.tokenizer.sep_token]            
            target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] * len(target_ids)
            padding_length = max_target_length - len(target_ids)
            target_ids += [self.tokenizer.pad_token_id] * padding_length
            target_mask += [0] * padding_length   
           
            features.append(InputFeatures(example_index, source_ids, target_ids, source_mask, target_mask))
        return features

class CodeBertSummDataModule(pl.LightningDataModule):
    def __init__(self, config):
        ''' CodeSearchNet data balanced for classification - from the CodeBERT repo. '''
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = CodeBertSummDataset(self.config, self.tokenizer, ttype='train')
            self.val_dataset = CodeBertSummDataset(self.config, self.tokenizer, ttype='dev')

        if stage == 'test' or stage == None:
            self.test_dataset = CodeBertSummDataset(self.config, self.tokenizer, ttype='test')

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))

    def val_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))

    def test_dataloader(self, batch_size=32):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n',' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n','')
            nl = ' '.join(nl.strip().split())            
            examples.append(Example(idx=idx, source=code, target=nl))

    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, example_id, source_ids, target_ids, source_mask, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
 
class Example(object):
    """A single training/test example."""
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target       

def preprocess_data(codesearch_path, dataset_path):
    def get_files(path):
        return [os.path.join(path, file) for file in os.listdir(path)]
    
    train, valid, test = get_files(codesearch_path + 'train/'), get_files(codesearch_path + 'valid/'), get_files(codesearch_path + 'test/')
    train_data, valid_data, test_data = {}, {}, {}
    
    for files, data in [[train, train_data], [valid, valid_data], [test, test_data]]:
        for file in files:
            f = pd.read_json(file, orient='records', compression='gzip', lines=True)

            for idx, js in f.iterrows():
                data[js['url']] = js

    for tag, data in [['train', train_data],['valid', valid_data],['test', test_data]]:
        with open('{}/{}.jsonl'.format(dataset_path, tag), 'w') as f, open("{}/{}.txt".format(dataset_path, tag)) as f1:
            for line in f1:
                line = line.strip()
                if line in data:
                    f.write(json.dumps(data[line]) + '\n')

if __name__ == '__main__':
    data_dir = "/content/drive/My Drive/Startup/data_files/data/python/"
    codesearch_path = os.path.join(data_dir, 'codesearch/')
    codebert_path = os.path.join(data_dir, 'codebert_summ/')

    preprocess_data(codesearch_path, codebert_path)