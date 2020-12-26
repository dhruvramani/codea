import os
import sys
import re
import pickle
import autopep8
import transformers

from transformers import AutoTokenizer, RobertaTokenizer

TAGS = {'eol' : '<EOL>\n', 'bof' : '<BOF>', 'eof': '<EOF>', 'tab': '<INDENT>',
        'comment': '<COMMENT>', 'num': '<NUM_LIT>', 'str': '<STR_LIT>',
        'main': '<STR_LIT:__main__>', 'post': '<STR_LIT:POST>',
        'get' : '<STR_LIT:GET>', }

def preprocess_code(config, code_block):
    # IMP - Maybe don't split it by new-line, just feed the whole corpus
    if config.prog_lang == 'python':
        code_block = autopep8.fix_code(code_block)

    code_block = TAGS['eol'].join([line for line in code_block.split('\n') if line.strip()]) 
    code_block = '{}\n{}\n{}'.format(TAGS['bof'], code_block, TAGS['eof'])

    if config.prog_lang == 'python':
        code_block = code_block.replace('    ', TAGS['tab']) 
        # ^ autopep8 replaces indentation w/ 4 spaces

        code_block = re.sub(r'#.*', TAGS['comment'], code_block)
        code_block = re.sub(r'\n\"\"\".*?\"\"\"', TAGS['comment'], code_block, flags=re.DOTALL)
        code_block = re.sub(r'\n\'\'\'.*?\'\'\'', TAGS['comment'], code_block, flags=re.DOTALL)

        code_block = re.sub(r'[-+]?[0-9]+', TAGS['num'], code_block)
        
        code_block = code_block.replace("\"__main__\"", TAGS['main'])
        code_block = code_block.replace("\"POST\"", TAGS['post'])
        code_block = code_block.replace("\"GET\"", TAGS['get'])
        code_block = re.sub(r"([bruf]*)(\"\"\"|'''|\"|')(?:(?!\2)(?:\\.|[^\\]))*\2", 
                            TAGS['str'], code_block, flags=re.DOTALL)

    return code_block

def get_tokenizer(config):
    # REFER for model changing - https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087
    if config.prog_lang == 'python':
        try :
            tokenizer = RobertaTokenizer.from_pretrained(config.tokenizer_path)
        except:
            print("UTILS - Creating new tokenizer.")
            tokenizer = RobertaTokenizer.from_pretrained('mrm8488/CodeBERTaPy')
            tokenizer.add_tokens(list(TAGS.values()))
            tokenizer.save_pretrained(config.tokenizer_path)
            
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