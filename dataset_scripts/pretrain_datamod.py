import torch
import numpy as np
import transformers
import pytorch_lightning as pl 

from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DataCollatorForSOP,
    DataCollatorForPermutationLanguageModeling)

import dataset_scripts.utils as utils

class PretrainDataCollate():
    def __init__(self, config, tokenizer, pretrain_tasks=['mlm']):
        self.config = config
        self.tokenizer = tokenizer

        tasks = {'mlm' : DataCollatorForLanguageModeling, 'wm' : DataCollatorForWholeWordMask, 
                 'plm' : DataCollatorForPermutationLanguageModeling, 'clm': DataCollatorWithPadding} #, 'sop' : DataCollatorForSOP}

        task_table = {'clm': {'count': 0, 'max_steps': 10},
                      'mlm': {'count': 0, 'max_steps': 10},
                      'plm': {'count': 0, 'max_steps': 10},
                      'wm' : {'count': 0, 'max_steps': 10}}

        assert set(pretrain_tasks).issubset(tasks.keys())
        self.pretrain_tasks = {key : tasks[key](self.tokenizer) for key in pretrain_tasks}
        self.task_stats = {key: task_table[key] for key in pretrain_tasks}

    def select_task(self):
        task_order = list(self.task_stats.keys())
        task_order.sort()    
        for idx, key in enumerate(task_order):
            if idx == len(task_order) - 1 and self.task_stats[key]['count'] == self.task_stats[key]['max_steps'] - 1:
                for count_key in task_order:
                    self.task_stats[count_key]['count'] = 0
                return self.pretrain_tasks[key] 
            
            if self.task_stats[key]['count'] < self.task_stats[key]['max_steps']:
                self.task_stats[key]['count'] += 1
                return self.pretrain_tasks[key]

    def collate_fn(self, batch):
        ''' - batch : A List of tokenized strings (tokenizer), type : List[Dict[Str : List]] '''
        task = self.select_task()
        return task(batch)

class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset, pretrain_tasks=['mlm', 'clm']):
        ''' - dataset : A uninialized torch Dataset object which returns tokenized string '''
        super().__init__()

        self.config = config
        self.tokenizer = utils.get_tokenizer(config)
        self.dataset = dataset(config, self.tokenizer)
        self.collator = PretrainDataCollate(config, self.tokenizer, pretrain_tasks)

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.dataset, batch_size=batch_size, collate_fn=self.collator.collate_fn)

if __name__ == '__main__':
    print("Testing pretrain_mod.py")
    from bigcode import BigCodeDataset
    
    config = utils.get_test_config(model='gpt2', dataset='bigcode')
    datamodule = PretrainDataModule(config, BigCodeDataset)
    datamodule.setup(stage='fit')
    train_loader = iter(datamodule.train_dataloader(batch_size=5))
    print(next(train_loader))
    print(next(train_loader))
    print(next(train_loader))
   