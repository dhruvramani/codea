import os 
import sys
import torch
import transformers 
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import get_config, BASE_DIR

# NOTE - A line completion system, for block level completion, use encoder-decoder model.

def train(config):
    datamodule = select_dataset(config, 'fit')
    model = select_model(config, datamodule)
    
    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    ckpt_callback = ModelCheckpoint(monitor='val_rouge2_fmeasure', dirpath=config.models_save_path, save_top_k=3)

    trainer = pl.Trainer(logger=logger, resume_from_checkpoint=config.resume_ckpt, callbacks=[ckpt_callback],
                tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus)
    
    trainer.fit(model, datamodule=datamodule)

def select_model(config, datamodule):
    if config.model == 'gpt2':
        from models import GPT2Code
        model = GPT2Code(config, tokenizer=datamodule.tokenizer)
    elif config.model == 'transfoxl':
        from models import TransXLCode
        model = TransXLCode(config, tokenizer=datamodule.tokenizer)
    else:
        raise NotImplementedError

    return model

def select_dataset(config, ttype):
    sys.path.append(BASE_DIR)
    
    if config.dataset == 'bigcode':
        from dataset_scripts import BigCodeDataModule
        datamodule = BigCodeDataModule(config)
    elif config.dataset == 'codesearch':
        from dataset_scripts import CodeSearchNetUnimodalDataModule
        datamodule = CodeSearchNetUnimodalDataModule(config)
    elif config.dataset == 'eth150':
        from dataset_scripts import ETH150DataModule
        datamodule = ETH150DataModule(config)
    elif config.dataset == 'all':
        from dataset_scripts import AllUnimodalDataModule
        datamodule = AllUnimodalDataModule(config, datasets=['bigcode', 'eth150', 'codesearch'])
    else:
        raise NotImplementedError
    
    datamodule.setup(stage=ttype)
    return datamodule

if __name__ == '__main__':
    config = get_config()
    train(config)