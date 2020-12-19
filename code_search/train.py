import os 
import sys
import torch
import transformers 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_config, BASE_DIR

def select_dataset(config, ttype):
    sys.path.append(BASE_DIR)
    
    if config.dataset == 'codesearch_bal':
        from dataset_scripts import CodeSearchBalancedDataModule
        datamodule = CodeSearchBalancedDataModule(config)
    elif config.dataset == 'codesearch':
        from dataset_scripts import CodeSearchNetMultimodalDataModule
        datamodule = CodeSearchNetMultimodalDataModule(config)
        
    datamodule.setup(stage=ttype)
    return datamodule

def select_model(config, datamodule):
    if config.model == 'p_codebert':
        from models import PretrainedCodeBERT
        total_steps = len(datamodule.train_dataloader(batch_size=config.batch_size)) // config.n_epochs
        model = PretrainedCodeBERT(config, total_steps, tokenizer=datamodule.tokenizer)

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