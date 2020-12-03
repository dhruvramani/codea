import os
import sys
import shutil
import pathlib
import datetime
import argparse

MODULE_DIR = os.path.dirname((os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(MODULE_DIR)
DATA_DIR = "/content/drive/My Drive/Startup/data/code_completion"
#DATA_DIR = "/scratch/sceatch2/dhruvramani/codecompletion_data"

TIME_STAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_config():
    parser = argparse.ArgumentParser("Code Completion - Model Independent Config.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'bart'])
    parser.add_argument('--prog_lang', type=str, default='python', choices=['python', 'java', 'javascript'])
    parser.add_argument('--exp_name', type=str, default='v0.0')

    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--auto_select_gpus', type=str2bool, default=True)

    # NOTE - Trainer args : not used yet - but very handy, use later.
    parser.add_argument('--auto_scale_batch_size', type=str, default='binsearch')
    parser.add_argument('--auto_lr_find', type=str2bool, default=False)

    # NOTE - See the modifications to paths after parsing below. 
    parser.add_argument('--data_path', type=str, default=DATA_DIR)
    parser.add_argument('--cache_path', type=str, default=DATA_DIR)
    parser.add_argument('--models_save_path', type=str, default=os.path.join(BASE_DIR, 'save/code_completion/models/'))
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(BASE_DIR, 'save/code_completion/tensorboard/'))

    parser.add_argument('--n_epochs', type=int, default=8)    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=200)

    config = parser.parse_args()
    
    config.data_path = os.path.join(config.data_path, '{}/data/{}/'.format(config.model, config.prog_lang))
    config.cache_path = os.path.join(config.data_path, '{}/cache/{}/'.format(config.model, config.prog_lang))
    config.models_save_path = os.path.join(config.models_save_path, '{}/{}_{}/'.format(config.model, config.prog_lang, config.exp_name)) 
    config.tensorboard_path = os.path.join(config.tensorboard_path, '{}/{}_{}/'.format(config.model, config.prog_lang, config.exp_name)) 

    create_dir(config.data_path, recreate=False)
    create_dir(config.cache_path, recreate=False)
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