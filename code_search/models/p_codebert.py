import torch
import transformers 
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

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

        self.val_acc = pl.metrics.Accuracy()
        self.val_f1 = pl.metrics.Fbeta(num_classes=self.num_labels, beta=1.0)
        self.test_acc = pl.metrics.Accuracy()
        self.test_f1 = pl.metrics.Fbeta(num_classes=self.num_labels, beta=1.0)
                
    def forward(self, nl_text, code):
        encoded_input = self.tokenizer(nl_text, code)
        outputs = self.model(**encoded_input)
        results = torch.softmax(outputs.logits, dim=1).tolist()
        return results

    def training_step(self, train_batch, batch_idx):
        inputs = {'input_ids': train_batch[0],
                  'attention_mask': train_batch[1],
                  'token_type_ids': None, # RoBERTa doesn’t have token_type_ids
                  'labels': train_batch[3]
                }
        outputs = self.model(**inputs)
        loss = outputs[0]
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs = {'input_ids': val_batch[0],
                  'attention_mask': val_batch[1],
                  'token_type_ids': None, # RoBERTa doesn’t have token_type_ids
                  'labels': val_batch[3]
                }
        outputs = self.model(**inputs)
        val_loss, logits = outputs[:2]

        acc = self.val_acc(logits, inputs['labels'])
        f1 = self.val_f1(logits, inputs['labels'])
        self.log_dict({'val_loss' : val_loss, 'val_acc' : acc, 'val_f1' : f1})
        return val_loss

    def test_step(self, test_batch, batch_idx):
        inputs = {'input_ids': test_batch[0],
                  'attention_mask': test_batch[1],
                  'token_type_ids': None, # RoBERTa doesn’t have token_type_ids
                  'labels': test_batch[3]
                }
        outputs = self.model(**inputs)
        logits = outputs[1]
        
        acc = self.test_acc(logits, inputs['labels'])
        f1 = self.test_f1(logits, inputs['labels'])
        self.log_dict({'test_acc' : acc, 'test_f1' : f1})

    def configure_optimizers(self, learning_rate=1e-5, eps=1e-8, weight_decay=0.0):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, self.config.warmup_steps, self.total_steps)
        return [optimizer], [lr_scheduler]
    
    def backward(self, loss, optimizer, optimizer_idx, max_grad_norm=1.0):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)