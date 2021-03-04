import torch
import transformers 
import pytorch_lightning as pl

from torch.nn import functional as F
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

from models.p_codebert_model import Seq2Seq, Beam

''' SOURCE https://github.com/microsoft/CodeBERT/blob/master/code2nl/run.py | model.py '''

class PretrainedCodeBERT(pl.LightningModule):
    def __init__(self, config, train_len, model_config=None, tokenizer=None):
        super(PretrainedCodeBERT, self).__init__()

        self.config = config
        self.train_len = train_len
        self.model_config = RobertaConfig.from_pretrained('microsoft/codebert-base') if model_config is None else model_config
        self.model_config.output_hidden_states = True

        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=True) if tokenizer is None \
                         else tokenizer

        encoder = RobertaModel(config=self.model_config) # .from_pretrained('microsoft/codebert-base', self.model_config)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=self.model_config.hidden_size, nhead=self.model_config.num_attention_heads)
        decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.model = Seq2Seq(encoder=encoder, decoder=decoder, config=self.model_config, beam_size=10, \
            max_length=128, sos_id=self.tokenizer.cls_token_id, eos_id=self.tokenizer.sep_token_id)
        
        # pretrained model link : https://drive.google.com/uc?id=1YrkwfM-0VBCJaa9NYaXUQPODdGPsmQY4
        # pretrained_path = 'code_summ_models/'.join([config.models_save_path.split("code_summ_models/")[0], 'pytorch_model.bin'])
        # print(pretrained_path)
        # self.model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

        self.metric1 = None
        self.metric2 = None
                
    def forward(self, input_ids, attention_mask):
        preds = self.model(source_ids=input_ids, source_mask=attention_mask)
        return preds

    def infer(self, code):
        # Source : https://github.com/graykode/ai-docstring/blob/master/server/server.ipynb
        enc_input = self.tokenizer(code, return_tensors='pt')
        preds = self.forward(enc_input['input_ids'], enc_input['attention_mask'])

        p = []
        for pred in preds:
            t = pred[0].cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[: t.index(0)]
            text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
            p.append(text)

        return p

    def _step(self, batch, batch_idx, return_logits=False, return_loss=True):
        if not(return_logits or return_loss):
            raise ReferenceError
        loss, logits = None, None
        if return_loss:
            loss, _, _ = self.model(source_ids=batch['input_ids'], source_mask=batch['attn_mask'], \
                target_ids=batch['labels'], target_mask=batch['label_attn_mask'])
        
        if return_logits:
            logits = self.model(source_ids=batch['input_ids'], source_mask=batch['attn_mask'])
        
        return loss, logits

    def training_step(self, train_batch, batch_idx):
        loss, _ = self._step(train_batch, batch_idx, return_loss=True)
        
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        val_loss, _ = self._step(val_batch, batch_idx, return_logits=True, return_loss=True)

        score = self.compute_metrics(logits, val_batch['target_ids'])
        score = {'val_' + key : score[key] for key in score}

        self.log('val_loss', val_loss)
        self.log_dict(score)
        return val_loss

    def test_step(self, test_batch, batch_idx):
        _, logits = self._step(test_batch, batch_idx, return_logits=True, return_loss=False)

        score = self.compute_metrics(logits, test_batch['target_ids'])
        score = {'test_' + key : score[key] for key in score}

        self.log_dict(score)

    def configure_optimizers(self, learning_rate=1e-3, eps=1e-8, weight_decay=0.0):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.train_len * self.config.n_epochs *0.1), \
                                                                              num_training_steps=self.train_len * self.config.n_epochs)
        return [optimizer], [lr_scheduler]
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def compute_metrics(pred_ids, label_ids):
        from datasets import load_metric

        self.metric1 = load_metric('bleu') if self.metric1 is None else self.metric1
        self.metric2 = load_metric('rouge') if self.metric2 is None else self.metric2

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