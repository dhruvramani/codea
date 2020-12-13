import torch
import numpy as np
import transformers
import pytorch_lightning as pl 


from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollatorWithPadding
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DataCollatorForSOP,
    DataCollatorForPermutationLanguageModeling)

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
        self.pretrain_tasks = [tasks[key](self.tokenizer) for key in pretrain_tasks]
        self.task_stats = {key: task_table[key] for key in pretrain_tasks}

    def select_task():
        task_order = list(self.task_stats.keys()).sort()        
        for idx, key in enumerate(task_order):
            if idx == len(task_order) - 1 and self.task_stats[key]['count'] == self.task_stats[key]['max_steps'] - 1:
                for count_key in enumerate(task_order):
                    self.task_stats[count_key]['count'] = 0
                return self.pretrain_tasks[key] 
            
            if self.task_stats[key]['count'] < self.task_stats[key]['max_steps']:
                self.task_stats[key]['count'] += 1
                return self.pretrain_tasks[key]

    def collate_fn(self, batch):
        ''' - batch : A List of tokenized strings (tokenizer), type : List[Dict[Str : List]] '''
        task = self.select_task()
        return task(batch)

# TODO - Have to write padding - might be there in tokenizer lib itself.

class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset, tokenizer, pretrain_tasks=['mlm']):
        ''' - dataset : A torch Dataset object which returns tokenized string '''
        super(PretrainDataModule, self).__init__()

        self.config = config
        self.dataset = dataset
        self.collator = PretrainDataCollate(config, tokenizer, pretrain_tasks)

    def train_dataloader(self, batch_size=None):
        return DataLoader(self.dataset, batch_size=batch_size, collate_fn=self.collator.collate_fn)
