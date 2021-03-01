import os
import sys
import shutil
import pathlib
import datetime
import argparse

MODULE_DIR = os.path.dirname((os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(MODULE_DIR)
DATA_DIR = "/content/drive/My Drive/Startup/data_files/"
SAVE_DIR  = "/content/drive/My Drive/Startup/save/"

TIME_STAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_config(def_model='p_codebert', def_plang='python', checkpoint=None,\
                save_dir=None, gdrive=True):
    save_dir = save_dir if save_dir is not None else SAVE_DIR
    parser = argparse.ArgumentParser("Code Summarization - Model Independent Config.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', type=str.lower, default=def_model, choices=['bart', 'mbart', 'p_codebert'])
    parser.add_argument('--prog_lang', type=str.lower, default=def_plang, choices=['python', 'java', 'javascript', 'c'])
    parser.add_argument('--dataset', type=str.lower, default='codebert_summ', choices=['codesearch', 'codebert_summ'])
    parser.add_argument('--exp_name', type=str, default='v0.0')

    parser.add_argument('--resume_ckpt', type=str, default=checkpoint)

    # NOTE - See lightning docs.
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--auto_select_gpus', type=str2bool, default=True)
    parser.add_argument('--precision', type=int, default=32)

    # NOTE - Lightning trainer args - not used yet - very handy tho, use later.
    parser.add_argument('--auto_scale_batch_size', type=str, default='binsearch')
    parser.add_argument('--auto_lr_find', type=str2bool, default=False)

    # NOTE - See the modifications to paths below.
    parser.add_argument('--data_path', type=str, default=os.path.join(DATA_DIR, 'data/'))
    parser.add_argument('--cache_path', type=str, default=os.path.join(DATA_DIR, 'cache/'))
    parser.add_argument('--tokenizer_path', type=str, default=os.path.join(DATA_DIR, 'cache/'))
    parser.add_argument('--models_save_path', type=str, default=os.path.join(save_dir, 'code_summ_models/'))
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(save_dir, 'code_summ_tensorboard/'))

    parser.add_argument('--n_epochs', type=int, default=8)    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=200)

    config = parser.parse_args()
    if grdive == True:
        if config.dataset != 'all':
            config.data_path = os.path.join(config.data_path, '{}/{}/'.format(config.prog_lang, config.dataset))
            config.cache_path = os.path.join(config.cache_path, '{}/{}/'.format(config.prog_lang, config.dataset))
        config.tokenizer_path = os.path.join(config.tokenizer_path, '{}/tokenizer/'.format(config.prog_lang))
        config.models_save_path = os.path.join(config.models_save_path, '{}/{}_{}/{}/'.format(config.prog_lang, config.model, config.dataset, config.exp_name)) 
        config.tensorboard_path = os.path.join(config.tensorboard_path, '{}/{}_{}/{}/'.format(config.prog_lang, config.model, config.dataset, config.exp_name)) 
    
    config.resume_ckpt = os.path.join(config.models_save_path, config.resume_ckpt) if config.resume_ckpt else None
    if config.resume_ckpt is not None and not(os.path.isfile(config.resume_ckpt)):
        print("=> Checkpoint doesn't exist.")
        config.resume_ckpt = None

    if grdive == True:
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