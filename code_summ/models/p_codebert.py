import copy
import torch
import transformers 
import pytorch_lightning as pl

from datasets import load_metric
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

''' SOURCE https://github.com/microsoft/CodeBERT/blob/master/code2nl/run.py | model.py '''

class PretrainedCodeBERT(pl.LightningModule):
    def __init__(self, config, model_config=None, tokenizer=None):
        super(PretrainedCodeBERT, self).__init__()

        self.config = config
        self.num_labels = 2
        self.model_config = RobertaConfig.from_pretrained('microsoft/codebert-base') if model_config is None else model_config

        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=True) if tokenizer is None \
                         else tokenizer

        encoder = RobertaModel.from_pretrained('microsoft/codebert-base', config=self.model_config)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=self.model_config.hidden_size, nhead=self.model_config.num_attention_heads)
        decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.model = Seq2Seq(encoder=encoder, decoder=decoder, config=self.model_config, beam_size=10, \
            max_length=128, sos_id=self.tokenizer.cls_token_id, eos_id=self.tokenizer.sep_token_id) # TODO - maybe change max_length 32

        self.metric1 = load_metric('bleu')
        self.metric2 = load_metric('rouge')
                
    def forward(self, nl_text, code):
        encoded_input = self.tokenizer(nl_text, code)
        outputs = self.model(**encoded_input)
        results = torch.softmax(outputs.logits, dim=1).tolist()
        return results

    def _step(batch, batch_idx, return_logits=False, return_loss=True):
        if not(return_logits or return_loss):
            raise ReferenceError
        loss, logits = None, None
        if return_loss:
            loss, _, _ = self.model(source_ids=batch['source_ids'], source_mask=batch['source_mask'], \
                target_ids=batch['target_ids'], target_mask=batch['target_mask'])
        
        if return_logits:
            logits = self.model(source_ids=batch['source_ids'], source_mask=batch['source_mask'])
        
        return loss, logits

    def training_step(self, train_batch, batch_idx):
        loss, _ = self._step(train_batch, batch_idx, return_loss=True)
        
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        val_loss, logits = self._step(val_batch, batch_idx, return_logits=True, return_loss=True)

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

    def configure_optimizers(self, learning_rate=5e-5, eps=1e-8, weight_decay=0.0):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 0, -1)
        return [optimizer], [lr_scheduler]
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

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

''' SOURCE https://github.com/microsoft/CodeBERT/blob/master/code2nl/model.py '''

class Seq2Seq(torch.nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = torch.nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not """
        if self.config.torchscript:
            first_module.weight = torch.nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        if target_ids is not None:  
            attn_mask =  -1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask, memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            #Predict 
            preds = []       
            zero = torch.LongTensor(1).fill_(0)     
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i+1]
                context_mask = source_mask[i:i+1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask = -1e4 *(1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask, memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp= beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[: self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))
                
            preds = torch.cat(preds, 0)                
            return preds

class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda if torch.cuda.is_available() else torch
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[: self.size - len(self.finished)]
        return self.finished[: self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[: timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[:: -1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence