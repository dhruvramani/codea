import os 
import sys
import torch
import transformers 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_config, BASE_DIR

sys.path.append(BASE_DIR)

def create_pretained_codebert(config, ttype):
    from models import PretrainedCodeBERT
    from datasets import CodeSearchBalancedDM

    tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/codebert-base')

    datamodule = CodeSearchBalancedDM(config, tokenizer)
    datamodule.setup(stage=ttype)
    total_steps = len(datamodule.train_dataloader(batch_size=config.batch_size)) // config.n_epochs

    model = PretrainedCodeBERT(config, total_steps, tokenizer=tokenizer)

    return model, datamodule

def select_model(config, ttype='fit'):
    model, datamodule = None, None
    if config.model == 'pretrained_codebert':
        model, datamodule = create_pretained_codebert(config, ttype)

    return model, datamodule

def train(config, model, datamodule):
    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    trainer = pl.Trainer(default_root_dir=config.models_save_path, weights_save_path=config.models_save_path, 
                        logger=logger, resume_from_checkpoint=config.resume_from_checkpoint, 
                        tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus)
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    config = get_config()
    model, datamodule = select_model(config)
    train(config, model, datamodule)
