import os 
import sys
import torch
import transformers 
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import get_config, BASE_DIR

def train(config):
    datamodule = select_dataset(config, 'fit')
    model = select_model(config, datamodule)
    
    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    ckpt_callback = ModelCheckpoint(monitor='val_f1', dirpath=config.models_save_path, save_top_k=3)
    resume_path = ckpt_callback.best_model_path if (config.resume_best_checkpoint and ckpt_callback.best_model_path != '') \
                    else None

    trainer = pl.Trainer(logger=logger, resume_from_checkpoint=resume_path, callbacks=[ckpt_callback],
                tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus)
    
    trainer.fit(model, datamodule=datamodule)

def select_model(config, datamodule):
    if config.model == 'p_codebert':
        from models import PretrainedCodeBERT
        total_steps = len(datamodule.train_dataloader(batch_size=config.batch_size)) // config.n_epochs
        model = PretrainedCodeBERT(config, total_steps, tokenizer=datamodule.tokenizer)
    else:
        raise NotImplementedError

    return model

def select_dataset(config, ttype):
    sys.path.append(BASE_DIR)
    
    if config.dataset == 'codesearch_bal':
        from dataset_scripts import CodeSearchBalancedDataModule
        datamodule = CodeSearchBalancedDataModule(config)
    else:
        raise NotImplementedError

    datamodule.setup(stage=ttype)
    return datamodule

if __name__ == '__main__':
    config = get_config()
    train(config)