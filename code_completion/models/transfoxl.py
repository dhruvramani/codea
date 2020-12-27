import torch
import transformers
import pytorch_lightning as pl

from datasets import load_metric
from torch.nn import functional as F
from transformers import TransfoXLConfig, TransfoXLTokenizer, TransfoXLLMHeadModel

class TransXLCode(pl.LightningModule):
    def __init__(self, config, model_config=None, tokenizer=None):
        super(TransXLCode, self).__init__()

        self.config = config
        self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103') if tokenizer is None \ 
                         else tokenizer
        self.model_config = TransfoXLConfig(vocab_size=self.tokenizer.get_vocab_size()) if model_config is None else model_config

        self.model = TransfoXLLMHeadModel
        self.model = self.model.from_pretrained('transfo-xl-wt103', config=self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.metric = load_metric('rouge')

    def forward(self, input_code, num_suggestions=5, num_beams=5, max_length=50):
        input_ids = self.tokenizer(input_code) # TODO - try out Top-K sampling [https://huggingface.co/blog/how-to-generate]
        gen_outputs = self.model.generate(input_ids, early_stopping=True, num_return_sequences=num_suggestions, \ 
            max_length=max_length, num_beams=num_beams)

        outputs = [self.tokenizer.decode(gen_op, skip_special_tokens=True) for gen_op in gen_outputs]
        return outputs

    def _step(self, batch, batch_idx):
        input_ids = batch['input_ids']

        outputs = self.model(**batch, labels=input_ids, return_dict=True) # See doc. for more info on Labels.
        return outputs

    def training_step(self, train_batch, batch_idx):
        outputs = self._step(train_batch, batch_idx)
        loss = outputs['loss']
        
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self._step(val_batch, batch_idx)

        loss, logits = outputs['loss'], outputs['logits']
        score = self.compute_metrics(logits, val_batch['input_ids'])
        score = {"val_" + key : score[key] for key in score}
        
        self.log('val_loss', loss)
        self.log_dict(score)

    def test_step(self, test_batch, batch_idx):
        outputs = self._step(test_batch, batch_idx)
        score = self.compute_metrics(outputs['logits'], test_batch['input_ids'])
        score = {"test_" + key : score[key] for key in score}
        self.log_dict(score)

    def compute_metrics(pred_ids, label_ids):
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        rouge_output = self.metric.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }