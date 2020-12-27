import os
import sys
import shutil
import pathlib
import datetime
import argparse

MODULE_DIR = os.path.dirname((os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(MODULE_DIR)
DATA_DIR = "/content/drive/My Drive/Startup/data_files/"
#DATA_DIR = "/scratch/sceatch2/dhruvramani/code_data/"
SAVE_DIR  = "/content/drive/My Drive/Startup/save/"

TIME_STAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_config():
    parser = argparse.ArgumentParser("Code Completion - Model Independent Config.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', type=str.lower, default='gpt2', choices=['gpt2', 'transfoxl', 'prophetnet']) 
    # NOTE - ^ change it to transformers model names
    parser.add_argument('--prog_lang', type=str.lower, default='python', choices=['python', 'java', 'javascript', 'c'])
    parser.add_argument('--dataset', type=str.lower, default='eth150', choices=['bigcode', 'codesearch', 'all', 'eth150'])
    parser.add_argument('--exp_name', type=str, default='v0.0')

    parser.add_argument('--resume_best_checkpoint', type=str2bool, default=1)

    # NOTE - See lightning docs.
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--auto_select_gpus', type=str2bool, default=True)

    # NOTE - Lightning trainer args - not used yet - very handy tho, use later.
    parser.add_argument('--auto_scale_batch_size', type=str, default='binsearch')
    parser.add_argument('--auto_lr_find', type=str2bool, default=False)

    # NOTE - See the modifications to paths below.
    parser.add_argument('--data_path', type=str, default=os.path.join(DATA_DIR, 'data/'))
    parser.add_argument('--cache_path', type=str, default=os.path.join(DATA_DIR, 'cache/'))
    parser.add_argument('--tokenizer_path', type=str, default=os.path.join(DATA_DIR, 'cache/'))
    parser.add_argument('--models_save_path', type=str, default=os.path.join(SAVE_DIR, 'code_completion_models/'))
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(SAVE_DIR, 'code_completion_tensorboard/'))

    parser.add_argument('--n_epochs', type=int, default=8)    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=200)

    config = parser.parse_args()
    if config.dataset != 'all':
        config.data_path = os.path.join(config.data_path, '{}/{}/'.format(config.prog_lang, config.dataset))
        config.cache_path = os.path.join(config.cache_path, '{}/{}/'.format(config.prog_lang, config.dataset))
    config.tokenizer_path = os.path.join(config.tokenizer_path, '{}/tokenizer/'.format(config.prog_lang))
    config.models_save_path = os.path.join(config.models_save_path, '{}/{}_{}/{}/'.format(config.prog_lang, config.model, config.dataset, config.exp_name)) 
    config.tensorboard_path = os.path.join(config.tensorboard_path, '{}/{}_{}/{}/'.format(config.prog_lang, config.model, config.dataset, config.exp_name)) 

    create_dir(config.data_path, recreate=False)
    create_dir(config.cache_path, recreate=False)
    create_dir(config.tokenizer_path, recreate=False)
    create_dir(config.models_save_path, recreate=False)
    create_dir(config.tensorboard_path, recreate=False)

    return config

def str2bool(string):
    return string.lower() == 'true'

def create_dir(path, recreate=False):
    if not os.path.isdir(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        print("# CREATED : {}.".format(path))
    elif recreate:
        shutil.rmtree(path)
        os.makedirs(path)
        print("# RE-CREATED : {}.".format(path))

if __name__ == '__main__':
    print("=> Testing config.py")
    config = get_config()
    print(config.models_save_path)