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
    ckpt_callback = ModelCheckpoint(monitor='val_bleu_score', dirpath=config.models_save_path, save_top_k=3)
    resume_path = ckpt_callback.best_model_path if (config.resume_best_checkpoint and ckpt_callback.best_model_path != '') \
                    else None

    trainer = pl.Trainer(logger=logger, resume_from_checkpoint=resume_path, callbacks=[ckpt_callback],
                tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus)
    
    trainer.fit(model, datamodule=datamodule)

def select_model(config, datamodule):
    if config.model == 'bart':
        from models import BartCode
        model = BartCode(config, tokenizer=datamodule.tokenizer)
    elif config.model == 'mbart':
        from models import MBartCode
        model = MBartCode(config, tokenizer=datamodule.tokenizer)
    else:
        raise NotImplementedError

    return model

def select_dataset(config, ttype):
    sys.path.append(BASE_DIR)
    
    if config.dataset == 'codesearch':
        from dataset_scripts import CodeSearchNetMultimodalDataModule
        datamodule = CodeSearchNetMultimodalDataModule(config)
    else:
        raise NotImplementedError
    
    datamodule.setup(stage=ttype)
    return datamodule

if __name__ == '__main__':
    config = get_config()
    train(config)