import torch
import transformers 
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel

from config import get_config
from dataset import get_codebert_search_dataloader

class PretrainedCodeBERT(pl.LightningModule):
    def __init__(self, config, total_steps, model_config=None, tokenizer=None):
        super(PretrainedCodeBERT, self).__init__()

        # total_steps calculated in train.py - needed for scheduler.

        self.config = config
        self.num_labels = 2
        self.total_steps = total_steps

        # IK, could've used ternary op - but code wasn't clean enough w/ that.
        self.model_config = model_config
        if self.model_config is None:
            self.model_config = AutoConfig.from_pretrained('microsoft/codebert-base', num_labels=self.num_labels, finetuning_task='codesearch')

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        self.model = AutoModel.from_pretrained('microsoft/codebert-base', config=self.model_config)

        self.val_acc = pl.metrics.Accuracy()
        self.val_f1 = pl.metrics.Fbeta(num_classes=self.num_labels, beta=1.0)
        self.test_acc = pl.metrics.Accuracy()
        self.test_f1 = pl.metrics.Fbeta(num_classes=self.num_labels, beta=1.0)
                
    def forward(self, nl_text, code):
        encoded_input = self.tokenizer(nl_text, code)
        outputs = self.model(**encoded_input)
        results = torch.softmax(outputs.logits, dim=1).tolist()
        return results

    def infer(self, batch):
        # NOTE/TODO - Implement your own data pipeline and change this
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': None, # RoBERTa doesn’t have token_type_ids
                  'labels': batch[3]
                }
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, train_batch, batch_idx):
        outputs = self.infer(train_batch)
        loss = outputs.loss
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self.infer(val_batch)
        val_loss = outputs.loss
        logits = outputs.logits

        acc = self.val_acc(logits, inputs['labels'])
        f1 = self.val_f1(logits, inputs['labels'])
        self.log_dict({'val_loss' : val_loss, 'val_acc' : acc, 'val_f1' : f1})
        return val_loss

    def test_step(self, test_batch, batch_idx):
        outputs = self.infer(test_batch)
        logits = outputs.logits
        
        acc = self.test_acc(logits, inputs['labels'])
        f1 = self.test_f1(logits, inputs['labels'])
        self.log_dict({'test_acc' : acc, 'test_f1' : f1})

    def configure_optimizers(self, learning_rate=3e-4, eps=1e-8, weight_decay=0.0):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, self.config.warmup_steps, self.total_steps)
        return [optimizer], [lr_scheduler]