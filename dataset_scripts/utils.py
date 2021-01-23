import os
import sys
import re
import glob
import shutil
import pickle
import autopep8
import torch
import transformers

from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer, RobertaTokenizer

MAX_LENS = {'gpt2': 1024, 'transfoxl': 1024}

SPECIAL_TAGS = {'bof' : '<BOF>', 'eof': '<EOF>'}
TAGS = {'eol' : '<NEWLINE>', 'tab': '<INDENT>', 'comment': '<COMMENT>', 'num': '<NUM_LIT>', 'str': '<STR_LIT>',}

PY_TAGS = {'<STR_LIT:__main__>': r"__main__", '<STR_LIT:POST>': r'post', '<STR_LIT:GET>': r'get', 
            '<STR_LIT:URL>': r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", 
            '<STR_LIT:PATH>': r'(.*)\/([^/]*)', '.split(<STR_LIT:SPLIT>)': r".split\(['\"](.*)['\"]\)", '<STR_LIT:FILEOP>': r'[rwa]{1}[b+]{0,2}',
            '<STR_LIT:UTF-8>': r'utf-8', '<STR_LIT:JSON>': r'\{.*\:\{.*\:.*\}\}', '<STR_LIT:RE>': r"r['\"](.*)['\"]", 
            '<STR_LIT:TRUE>': r'True', '<STR_LIT:FALSE>': r'False', '<STR_LIT:PKG>': r'[A-Za-z]*\.[A-Za-z.]+',
            '<NUM_LIT:SCI>': r'[-]*\d+e[-]*\d*', '<NUM_LIT:BOOL>': r'[01]{1}'}

def preprocess_code(config, code_block, nlines=True):
    # NOTE IMP - Maybe don't split it by new-line, just feed the whole corpus
    # NOTE - See bos_token, eos_token from https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer
    if config.prog_lang == 'python':
        code_block = autopep8.fix_code(code_block)
        code_block = code_block.replace('    ', TAGS['tab']) 
        # ^ autopep8 replaces indentation w/ 4 spaces

        code_block = re.sub(r'#[\s\S]*?(?=\n(?!#).*|$)', TAGS['comment'], code_block) #captures multiline-single comments too
        code_block = re.sub(r'\n\"\"\".*?\"\"\"', TAGS['comment'], code_block, flags=re.DOTALL)
        code_block = re.sub(r'\n\'\'\'.*?\'\'\'', TAGS['comment'], code_block, flags=re.DOTALL)
        
        for tag, rexp in PY_TAGS.items():
            if 'STR' in tag and ('SPLIT' not in tag) and ('RE' not in tag): # dirty HACK
                rexp = r"['\"]" + rexp + r"['\"]"
            code_block = re.sub(rexp, tag, code_block, flags=re.IGNORECASE)

        code_block = re.sub(r'[+-]?([0-9]*[.])?[0-9]+', TAGS['num'], code_block)
        code_block = re.sub(r"([bruf]*)(\"\"\"|'''|\"|')(?:(?!\2)(?:\\.|[^\\]))*\2", 
                            TAGS['str'], code_block, flags=re.DOTALL)

    eol_tag = TAGS['eol'] + '\n' if nlines else TAGS['eol']
    code_block = eol_tag.join([line for line in code_block.split('\n') if line.strip()]) 

    bof_tag, eof_tag = SPECIAL_TAGS['bof'], SPECIAL_TAGS['eof']
    
    code_block = '{}{}{}'.format(bof_tag, code_block, eof_tag)
    return code_block

def group_texts(tokenized_texts, block_size):
    # Source : https://github.com/huggingface/transformers/blob/ef93a254275c8d79a964564202979a169599f96d/examples/language-modeling/run_clm.py#L275
    #concatenated_tokenized_texts = {k: sum(tokenized_texts[k], []) for k in tokenized_texts.keys()}
    total_length = len(tokenized_texts[list(tokenized_texts.keys())[0]])
    if total_length < block_size:
        return {k: [t] for k, t in tokenized_texts.items()}
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in tokenized_texts.items()
    }
    return result

def get_tokenizer(config):
    # REFER for model changing - https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087
    if config.prog_lang == 'python':
        try :
            vocab_file = os.path.join(config.tokenizer_path, 'vocab.json')
            merges_file = os.path.join(config.tokenizer_path, 'merges.txt')
            tokenizer = RobertaTokenizer.from_pretrained(config.tokenizer_path)
        except:
            print("D UTILS : Creating new tokenizer.")
            tokenizer = RobertaTokenizer.from_pretrained('mrm8488/CodeBERTaPy', eos_token=SPECIAL_TAGS['eof'], bos_token=SPECIAL_TAGS['bof'])
            tokenizer.add_tokens(list(SPECIAL_TAGS.values()))
            tokenizer.add_tokens(list(TAGS.values()))
            tokenizer.add_tokens(list(PY_TAGS.keys()))
            tokenizer.save_pretrained(config.tokenizer_path)
            eofbof_ids = tokenizer.convert_tokens_to_ids([SPECIAL_TAGS['eof'], SPECIAL_TAGS['bof']])
            torch.save(eofbof_ids, f'{config.tokenizer_path}eofbof_ids.pt')
            
    elif config.prog_lang == 'javascript':
        tokenizer = AutoTokenizer.from_pretrained('mrm8488/codeBERTaJS')

    return tokenizer

def get_model_config(config, tokenizer):
    raise NotImplementedError

def get_test_config(model, dataset):
    import pathlib
    import argparse
    
    sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default=model)
    parser.add_argument('--prog_lang', type=str.lower, default='python')
    parser.add_argument('--dataset', type=str.lower, default=dataset)
    parser.add_argument('--data_path', type=str, default='/content/drive/My Drive/Startup/data_files/data/')
    parser.add_argument('--cache_path', type=str, default='/content/drive/My Drive/Startup/data_files/cache/')
    parser.add_argument('--tokenizer_path', type=str, default='/content/drive/My Drive/Startup/data_files/cache/')
    parser.add_argument('--batch_size', type=int, default=32)

    config = parser.parse_args()
    if config.dataset != 'all':
        config.data_path = os.path.join(config.data_path, '{}/{}/'.format(config.prog_lang, config.dataset))
        config.cache_path = os.path.join(config.cache_path, '{}/{}/'.format(config.prog_lang, config.dataset))
    config.tokenizer_path = os.path.join(config.tokenizer_path, '{}/tokenizer/'.format(config.prog_lang))

    pathlib.Path(config.data_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.cache_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.tokenizer_path).mkdir(parents=True, exist_ok=True)
    
    return config

def change_subdir_sys(path):
    # Changes the sub-directory structure to avoid Colab timeouts.
    from google.colab import drive
    from string import digits, ascii_uppercase, ascii_lowercase

    drive.mount("/content/drive")
    chars = digits + ascii_lowercase + ascii_uppercase 

    for char in chars:
        files = glob.glob(os.path.join(path, f'{char}*'))
        dest = os.path.join(path, f'_{char}/')
        os.mkdir(dest)
        for f in files:
            shutil.move(f, dest)
        print(dest)

    drive.flush_and_unmount()

'''
def index_file(path, cache_path):
    offsets = [0]
    cache_path = cache_path + "-index.pkl"
    
    if not os.path.isfile(cache_path):
        with open(path, 'rb') as f:
            while len(f.readline()) != 0:
                offsets.append(f.tell())

        del offsets[-1]
        with open(cache_path, 'wb') as f:
            pickle.dump(offsets, f)
    else:
        with open(cache_path, 'rb') as f:
            offsets = pickle.load(f)
    
    return offsets, len(offsets)
'''