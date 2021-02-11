import os 
import sys
import torch
import transformers 
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from config import get_config, BASE_DIR

# NOTE - A line completion system, for block level completion, use encoder-decoder model.

def train(config):
    datamodule = select_dataset(config, 'fit')
    model = select_model(config, datamodule.tokenizer)
    
    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    sv_cback = CheckpointEveryNSteps(config.val_check_interval + 1, config.models_save_path, ckpt_name="model.ckpt")
    callbacks = [sv_cback]
    if config.early_stopping:
        es_cback = EarlyStopping('val_rouge2_fmeasure')
        callbacks.append(es_cback)
        
    trainer = pl.Trainer(logger=logger, resume_from_checkpoint=config.resume_ckpt, callbacks=callbacks, precision=config.precision, limit_val_batches=100,
                tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus, val_check_interval=config.val_check_interval)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(os.path.join(config.models_save_path, "model.ckpt"))

def select_model(config, tokenizer):
    if config.model == 'gpt2':
        from models import GPT2Code
        model = GPT2Code(config, tokenizer=tokenizer)
    elif config.model == 'transfoxl':
        from models import TransXLCode
        model = TransXLCode(config, tokenizer=tokenizer)
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

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self, save_step_frequency, dirpath, ckpt_name='model.ckpt'):
        self.save_step_frequency = save_step_frequency
        self.path = os.path.join(dirpath, ckpt_name)

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            trainer.save_checkpoint(self.path)

if __name__ == '__main__':
    config = get_config()
    train(config)