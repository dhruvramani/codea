import os
import torch
import transformers
import pytorch_lightning as pl

from torch.nn import functional as F
from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel

class GPT2Code(pl.LightningModule):
    def __init__(self, config, tokenizer, model_config=None):
        super(GPT2Code, self).__init__()

        self.config = config
        self.tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2') if tokenizer is None \
                         else tokenizer
        eofbof_ids = torch.load(os.path.join(config.tokenizer_path, 'eofbof_ids.pt'))
        self.model_config = GPT2Config.from_pretrained('distilgpt2') if model_config is None else model_config
        self.model_config.eos_token_id = eofbof_ids[0]
        self.model_config.bos_token_id = eofbof_ids[1]

        self.model = GPT2LMHeadModel
        # Loading works properly - see https://github.com/PyTorchLightning/pytorch-lightning/issues/3096#issuecomment-681065813
        self.model = self.model.from_pretrained('distilgpt2', config=self.model_config) if self.config.resume_ckpt is None\
                        else self.model(config=self.model_config)
        
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.metric = None

    def forward(self, input_ids, num_suggestions=5, num_beams=5, max_length=50):
        # TODO - try out pipeline OR Top-K sampling [https://huggingface.co/blog/how-to-generate]
        gen_outputs = self.model.generate(input_ids, early_stopping=True, num_return_sequences=num_suggestions, \
            max_length=max_length, num_beams=num_beams)

        return gen_outputs

    def infer(self, input_code):
        tokens = self.tokenizer(input_code) 
        gen_outputs = self.forward(tokens['input_ids'])
        outputs = [self.tokenizer.decode(gen_op, skip_special_tokens=True) for gen_op in gen_outputs]
        return outputs

    def training_step(self, train_batch, batch_idx):
        input_ids = train_batch['input_ids']
        loss = self.model(**train_batch, labels=input_ids, return_dict=True)['loss'] 
        # See doc. for more info on Labels.
        
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input_ids = val_batch['input_ids']
        outputs = self.model(**val_batch, labels=input_ids, return_dict=True)
        loss = outputs['loss']
        logits = torch.argmax(outputs['logits'], dim=-1)
        del outputs

        score = self.compute_metrics(logits, val_batch['input_ids'])
        score = {"val_" + key : score[key] for key in score}
        
        self.log('val_loss', loss)
        self.log_dict(score)

    def test_step(self, test_batch, batch_idx):
        input_ids = test_batch['input_ids']
        logits = self.model(**test_batch, labels=input_ids, return_dict=True)['logits']
        logits = torch.argmax(logits, dim=-1)

        score = self.compute_metrics(logits, test_batch['input_ids'])
        score = {"test_" + key : score[key] for key in score}

        self.log_dict(score)
    
    def configure_optimizers(self, learning_rate=1e-5):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        return optimizer

    def compute_metrics(self, pred_ids, label_ids):
        from datasets import load_metric

        self.metric = load_metric('rouge') if self.metric is None else self.metric

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        rouge_output = self.metric.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }