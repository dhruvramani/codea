import os 
import sys
import torch
import transformers 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_config, BASE_DIR

def select_dataset(config, ttype):
    sys.path.append(BASE_DIR)
    
    if config.dataset == 'bigcode':
        from dataset_scripts import BigCodeDataModule
        datamodule = BigCodeDataModule(config)
    elif config.dataset == 'codesearch':
        from dataset_scripts import CodeSearchNetUnimodalDataModule
        datamodule = CodeSearchNetUnimodalDataModule(config)
    
    datamodule.setup(stage=ttype)
    return datamodule

def select_model(config, datamodule):
    if config.model == 'gpt2':
        from models import GPT2Code
        model = GPT2Code(config, tokenizer=datamodule.tokenizer)
    elif config.model == 'transfoxl':
        from models import TransXLCode
        model = TransXLCode(config, tokenizer=datamodule.tokenizer)

    return model

def train(config):
    datamodule = select_dataset(config, 'fit')
    model = select_model(config, datamodule)
    
    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    trainer = pl.Trainer(default_root_dir=config.models_save_path, weights_save_path=config.models_save_path, 
                        logger=logger, resume_from_checkpoint=config.resume_from_checkpoint, 
                        tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus)
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    config = get_config()
    train(config)