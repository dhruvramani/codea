import torch
import transformers 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_config
from train import select_model

def test(config, model, datamodule):
    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    trainer = pl.Trainer(default_root_dir=config.models_save_path, weights_save_path=config.models_save_path, logger=logger)
    trainer.test(model=model, datamodule=datamodule)

if __name__ == '__main__':
    config = get_config()
    model, datamodule = select_model(config)
    test(config, model, datamodule)
