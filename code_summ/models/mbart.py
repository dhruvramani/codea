import torch
import transformers
import pytorch_lightning as pl
from torch.nn import functional as F

from datasets import load_metric
from transformers.modeling_bart import shift_tokens_right
from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from transformers import MBartConfig, MBartTokenizer, MBartForConditionalGeneration

class MBartCode(pl.LightningModule):
    def __init__(self, config, model_config, code_tokenizer):
        super(MBartCode, self).__init__()
        ''' NOTE - All 3 Barts use the same config and tokenizer. See logic in the notes. '''

        self.config = config
        self.code_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base') if code_tokenizer is None else code_tokenizer
        self.model_config = BartConfig() if model_config is None else model_config
        self.eng_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
        self.code_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base', config=self.model_config)
        self.eng_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base', config=self.model_config)
        self.multi_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base', config=self.model_config)

        self.code_bart.resize_token_embeddings(len(self.code_tokenizer))
        self.multi_bart.resize_token_embeddings(len(self.code_tokenizer))

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
        input_id = self.code_tokenizer(input_code)
        summary_id = self.multi_bart.generate(input_id, num_beams=num_beams, max_length=max_length, early_stopping=True)
        output = self.eng_tokenizer.decode(summary_id, skip_special_tokens=True)
        return output

    def _step(self, batch, batch_idx):
        # TODO - Training works for multi & uni. Have to make a dataset for both multi & uni.
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['labels'] if self.current_train == 'multi' else batch['input_ids']

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