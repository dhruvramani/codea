import os
import sys
import gzip
import json
import numpy as np
from more_itertools import chunked

import torch
import pytorch_lightning as pl 

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

''' SOURCE https://github.com/microsoft/CodeBERT/blob/master/codesearch/utils.py '''

class CodeSearchBalancedDataset(Dataset):
    def __init__(self, config, tokenizer, ttype):
        self.config = config
        self.tokenizer = tokenizer

        self.processor = CodesearchProcessor()
        self.files = {'train' : 'train.txt', 'dev' : 'valid.txt', 'test' : 'test/batch_0.txt'}
        self.get_examples = {'train' : self.processor.get_train_examples, 'dev' : self.processor.get_dev_examples, 'test' : self.processor.get_test_examples}
        self.cache_dir = os.path.join(self.config.cache_path, 'cached_{}_{}_{}/'.format(ttype, self.files[ttype].split('.')[0], self.config.max_seq_length))
        
        self.cache_len = 1000
        self.prev_cache_idx = -1
        self.cache = None

        self._setup(ttype)

    def _setup(self, ttype='train'):
        cache_contents = os.listdir(self.cache_dir)
        
        if cache_contents != []:
            self.len = torch.load(os.path.join(self.cache_dir, 'len'))
        else:
            print("D CS-B - Creating cache.")
            label_list = self.processor.get_labels()
            examples = self.get_examples[ttype](self.config.data_path, self.files[ttype])
            if ttype == 'test':
                examples, _ = examples

            features = self.convert_examples_to_features(examples, label_list, self.config.max_seq_length, cls_token=self.tokenizer.cls_token, sep_token=self.tokenizer.sep_token)
            self.len = len(features)

            torch.save(self.len, os.path.join(self.cache_dir, 'len'))

            j = 0
            for i in range(self.len // self.cache_len):
                cached_features_file = os.path.join(self.cache_dir, f'{i}.pt')
                torch.save(features[i * self.cache_len : (i+1) * self.cache_len], cached_features_file)
                j = i + 1

            if j * self.cache_len < self.len:
                cached_features_file = os.path.join(self.cache_dir, f'{j}.pt')
                torch.save(features[j * self.cache_len :], cached_features_file)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cache_idx = idx // self.cache_len
        if cache_idx != self.prev_cache_idx:
            cache_file = os.path.join(self.cache_dir, f'{cache_idx}.pt')
            self.cache = torch.load(cache_file)
            self.prev_cache_idx = cache_idx

        content = self.cache[idx % self.cache_len]
        content = (content.input_ids, content.input_mask, content.segment_ids, content.label_id)
        content = (torch.LongTensor(c) for c in content)
        
        return content  

    def convert_examples_to_features(self, examples, label_list, max_seq_length,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)[:50]
            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]
            '''
                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
            '''
            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]

            # if ex_index < 5:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join(
            #         [str(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                              segment_ids=segment_ids, label_id=label_id))
        return features

class CodeSearchBalancedDataModule(pl.LightningDataModule):
    def __init__(self, config):
        ''' CodeSearchNet data balanced for classification - from the CodeBERT repo. '''
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            self.train_dataset = CodeSearchBalancedDataset(self.config, self.tokenizer, ttype='train')
            self.val_dataset = CodeSearchBalancedDataset(self.config, self.tokenizer, ttype='dev')

        if stage == 'test' or stage == None:
            self.test_dataset = CodeSearchBalancedDataset(self.config, self.tokenizer, ttype='test')

    def train_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size=None):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size)

    def test_dataloader(self, batch_size=32):
        batch_size = self.config.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class CodesearchProcessor(object):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """Gets a collection of `InputExample`s for the train set."""        
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """Gets the list of labels for this data set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if (set_type == 'test'):
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def download_dataset(config):
    import gdown
    import zipfile

    url = 'https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo'
    dataset_path = '/'.join(config.data_path.split('/')[:-1])
    dir_path = '{}/codesearch_data'.format(dataset_path)
    dataset_path = '{}/codesearch_data.zip'.format(dataset_path)
    print("File ", dataset_path, "\nStarting download")
    gdown.download(url, dataset_path, quiet=False)

    with zipfile.ZipFile(dataset_path ,"r") as zip_ref:
        zip_ref.extractall(dir_path)
    
    print("Downloaded.")

def preprocess_test_data(config, test_batch_size=self.cache_len):
    def format_str(string):
        for char in ['\r\n', '\r', '\n']:
            string = string.replace(char, ' ')
        return string

    path = os.path.join(config.data_path, '{}_test_0.jsonl.gz'.format(config.prog_lang))
    print(path)
    with gzip.open(path, 'r') as pf:
        data = pf.readlines()  

    idxs = np.arange(len(data))
    data = np.array(data, dtype=np.object)

    np.random.seed(0)   # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    data = data[idxs]
    batched_data = chunked(data, test_batch_size)

    print("start processing")
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break # the last batch is smaller than the others, exclude.
        examples = []
        for d_idx, d in enumerate(batch_data): 
            line_a = json.loads(str(d, encoding='utf-8'))
            doc_token = ' '.join(line_a['docstring_tokens'])
            for dd in batch_data:
                line_b = json.loads(str(dd, encoding='utf-8'))
                code_token = ' '.join([format_str(token) for token in line_b['code_tokens']])

                example = (str(1), line_a['url'], line_b['url'], doc_token, code_token)
                example = '<CODESPLIT>'.join(example)
                examples.append(example)

        data_path = os.path.join(config.data_path, 'test/')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, 'batch_{}.txt'.format(batch_idx))
        print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))