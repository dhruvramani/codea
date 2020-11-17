import os
import sys
import shutil
import pathlib
import argparse

MODULE_DIR = os.path.dirname((os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(MODULE_DIR)
DATA_DIR = "/content/drive/My Drive/Startup/data/nl_code_search"

TIME_STAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_config():
    parser = argparse.ArgumentParser("NL Code Search",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--exp_name', type=str, default='v0.0')
    parser.add_argument('--model', type=str, default='x', choices=['x', 'y'])

    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--is_train', type=str2bool, default=False)
    parser.add_argument('--seed', '-s', type=int, default=0)

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_graphs', type=str2bool, default=False)
    parser.add_argument('--save_interval_steps', type=int, default=100)

    parser.add_argument('--data_path', type=str, default=os.path.join(DATA_DIR, 'code_search_net/'))
    parser.add_argument('--models_save_path', type=str, default=os.path.join(MODULE_DIR, 'save/models/'))
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(MODULE_DIR, 'save/tensorboard/'))
    
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=25) #50
    parser.add_argument('--max_sequence_length', type=int, default=32)

    config = parser.parse_args()
    
    config.models_save_path = os.path.join(config.models_save_path, '{}_{}/'.format(config.model, config.exp_name)) 
    config.tensorboard_path = os.path.join(config.tensorboard_path, '{}_{}/'.format(config.model, config.exp_name)) 

    recreate = config.is_train and not config.resume
    create_dir(config.models_save_path, recreate=recreate)
    create_dir(config.tensorboard_path, recreate=recreate)

    return config

def str2bool(string):
    return string.lower() == 'true'

def create_dir(path, recreate=False):
    if not os.path.isdir(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        print("=> CREATED : {}.".format(path))
    else recreate:
        shutil.rmtree(path)
        os.makedirs(path)
        print("=> RE-CREATED : {}.".format(path))
