import torch
import transformers 
import pytorch_lightning as pl

from torch.nn import functional as F
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

''' SOURCE https://github.com/microsoft/CodeBERT/blob/master/codesearch/run_classifier.py '''

class PretrainedCodeBERT(pl.LightningModule):
    def __init__(self, config, total_steps, model_config=None, tokenizer=None):
        super(PretrainedCodeBERT, self).__init__()

        self.config = config
        self.num_labels = 2
        self.total_steps = total_steps # total_steps calculated in train.py - needed for scheduler.
        self.model_config = RobertaConfig.from_pretrained('microsoft/codebert-base', num_labels=self.num_labels, finetuning_task='codesearch') \
                            if model_config is None else model_config

        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base') if tokenizer is None \
                         else tokenizer

        self.model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', config=self.model_config)

        self.acc = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1(num_classes=self.num_labels)
                
    def forward(self, nl_text, code):
        encoded_input = self.tokenizer(nl_text, code)
        outputs = self.model(**encoded_input)
        results = torch.softmax(outputs.logits, dim=1).tolist()
        return results

    def _step(self, batch, batch_idx):
        inputs = {'input_ids': batch['input_ids'],
          'attention_mask': batch['attn_mask'],
          'token_type_ids': None, # RoBERTa doesn’t have token_type_ids
          'labels': batch['labels']
        }
        outputs = self.model(**inputs, return_dict=True)

        return outputs        

    def training_step(self, train_batch, batch_idx):
        outputs = self._step(train_batch, batch_idx)
        loss = outputs['loss']

        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self._step(val_batch, batch_idx)
        val_loss, logits = outputs['loss'], outputs['logits']

        labels = val_batch['labels']
        score = self.compute_metrics(logits, labels)
        score = {'val_' + key : score[key] for key in score}

        self.log('val_loss', val_loss)
        self.log_dict(score)
        return val_loss

    def test_step(self, test_batch, batch_idx):
        outputs = self._step(test_batch, batch_idx)
        logits = outputs['logits']
        
        labels = test_batch['labels']
        score = self.compute_metrics(logits, labels)
        score = {'test_' + key : score[key] for key in score}

        self.log_dict(score)

    def configure_optimizers(self, learning_rate=1e-5, eps=1e-8, weight_decay=0.0):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 0, self.total_steps)
        return [optimizer], [lr_scheduler]
    
    def backward(self, loss, optimizer, optimizer_idx, max_grad_norm=1.0):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

    def compute_metrics(self, logits, labels):
        acc = self.acc(logits, labels)
        f1 = self.f1(logits, labels)

        return {'acc' : acc, 'f1' : f1}