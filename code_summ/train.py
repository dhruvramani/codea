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
    #ckpt_callback = ModelCheckpoint(monitor='val_bleu_score', dirpath=config.models_save_path, save_top_k=3)

    trainer = pl.Trainer(logger=logger, resume_from_checkpoint=config.resume_ckpt, num_sanity_val_steps=0, #callbacks=[ckpt_callback],
                tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus, precision=config.precision)
    
    trainer.fit(model, datamodule=datamodule)

def select_model(config, datamodule):
    if config.model == 'bart':
        from models import BartCode
        model = BartCode(config, tokenizer=datamodule.tokenizer)
    elif config.model == 'mbart':
        from models import MBartCode
        model = MBartCode(config, tokenizer=datamodule.tokenizer)
    elif config.model == 'p_codebert':
        from models import PretrainedCodeBERT
        train_len = len(datamodule.train_dataloader())
        model = PretrainedCodeBERT(config, train_len, tokenizer=datamodule.tokenizer)
    else:
        raise NotImplementedError

    return model

def select_dataset(config, ttype='fit'):
    sys.path.append(BASE_DIR)
    
    if config.dataset == 'codesearch':
        from dataset_scripts import CodeSearchNetMultimodalDataModule
        datamodule = CodeSearchNetMultimodalDataModule(config)
    elif config.dataset == 'codebert_summ':
        from dataset_scripts import CodeBertSummDataModule
        datamodule = CodeBertSummDataModule(config)
    else:
        raise NotImplementedError
    
    datamodule.setup(stage=ttype)
    return datamodule

if __name__ == '__main__':
    config = get_config()
    train(config)