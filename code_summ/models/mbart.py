import torch
import transformers
import pytorch_lightning as pl
from torch.nn import functional as F

from datasets import load_metric
from transformers.modeling_bart import shift_tokens_right
from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from transformers import MBartConfig, MBartTokenizer, MBartForConditionalGeneration

class MBartCode(pl.LightningModule):
    def __init__(self, config, model_config, tokenizer, eng_config=None, eng_tokenizer=None):
        super(MBartCode, self).__init__()
        ''' NOTE - All 3 Barts use the same config and tokenizer. See logic in the notes. '''

        self.config = config
        self.eng_pretrained_name = 'facebook/bart-base' # 'facebook/bart-large'
        # TODO : edit this later - not clean.
        self.pretrained = model_config is None and tokenizer is None

        self.model_config = BartConfig() if model_config is None else model_config
        self.tokenizer = BartTokenizer.from_pretrained(self.pretrained_name) if tokenizer is None else tokenizer
        
        self.eng_bart = BartForConditionalGeneration
        self.eng_bart = self.eng_bart.from_pretrained(self.pretrained_name, config=self.model_config) if self.pretrained \
                     else self.eng_bart(self.model_config)

        self.code_bart = BartForConditionalGeneration(self.model_config)
        self.multi_bart = BartForConditionalGeneration(self.model_config)
        self.multi_bart.encoder = self.code_bart.encoder
        self.multi_bart.decoder = self.eng_bart.decoder

        self.current_train = 'code'
        self.current_model = self.code_bart

        self.metric1 = load_metric('bleu')
        self.metric2 = load_metric('rouge')

    def switch_model(self, choice):
        switches = {'eng' : self.eng_bart, 'code' : self.code_bart, 'multi' : self.multi_bart}
        assert choice.lower() is in switches.keys()

        self.current_train = choice.lower()
        self.current_model = switches[self.current_train]

    def forward(self, input_code, num_beams=5, max_length=50):
        input_id = self.tokenizer(input_code)
        summary_id = self.multi_bart.generate(input_id, num_beams=num_beams, max_length=max_length, early_stopping=True)
        output = self.tokenizer.decode(summary_id, skip_special_tokens=True)
        return output

    def _step(self, batch, batch_idx):
        # Multi-Modal and Uni-Modal data have different formats. Refer bart.py
        if self.current_train == 'multi':
            inputs, targets = batch['code'], batch['summary']
            input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            labels = targets['input_ids']
        else:
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            labels = batch['input_ids']

        decoder_input_ids = shift_tokens_right(labels, self.tokenizer.pad_token_id)
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # NOTE - not sure if this is needed.

        outputs = self.model(input_ids, attention_mask, decoder_input_ids=decoder_input_ids\
                  labels=labels, return_dict=True)
        return outputs

    def training_step(self, train_batch, batch_idx):
        outputs = self._step(train_batch, batch_idx)
        loss = outputs['loss']

        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self._step(val_batch, batch_idx)

        loss, logits = outputs['loss'], outputs['logits']
        score = self.compute_metrics(logits, val_batch['summary']['input_ids'])
        score = {"val_" + key : score[key] for key in score}
        
        self.log('val_loss', loss)
        self.log_dict(score)

    def test_step(self, test_batch, batch_idx):
        outputs = self._step(test_batch, batch_idx)
        score = self.compute_metrics(outputs['logits'], test_batch['summary']['input_ids'])
        score = {"test_" + key : score[key] for key in score}

        self.log_dict(score)

    def compute_metrics(pred_ids, label_ids):
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        bleu_output = self.metric1.compute(predictions=pred_str, references=label_str)
        rouge_output = self.metric2.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "bleu_score":  round(bleu_output.bleu, 4),
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }