import os
import sys
import re
import pickle
import transformers

def get_tokenizer(lang):
    if lang == 'python':
        tokenizer = transformers.AutoTokenizer('mrm8488/CodeBERTaPy')
    elif lang == 'javascript':
        tokenizer = transformers.AutoTokenizer('mrm8488/codeBERTaJS')

    return tokenizer

def get_model_config(config, tokenizer):
    raise NotImplementedError

def preprocess_code(config, code_block):
    # Follow intellicode to put <COMMENT>, <NUM> etc. in the line of code using re

    comment_tag = ''#'<COMMENT>'
    code_block = re.sub(r'#.*\n', comment_tag, code_block)
    code_block = re.sub(r'""".*?"""', comment_tag, code_block, flags=re.DOTALL)

    return code_block

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

def get_test_config(model, dataset):
    import pathlib
    import argparse
    
    sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str.lower, default=model)
    parser.add_argument('--prog_lang', type=str.lower, default='python')
    parser.add_argument('--dataset', type=str.lower, default=dataset)
    parser.add_argument('--data_path', type=str, default='/content/drive/My Drive/Startup/data_files/data/')
    parser.add_argument('--batch_size', type=int, default=32)

    config = parser.parse_args()
    config.data_path = os.path.join(config.data_path, '{}/{}/'.format(config.prog_lang, config.dataset))
    pathlib.Path(config.data_path).mkdir(parents=True, exist_ok=True)

    return config