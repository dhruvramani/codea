import torch
import transformers 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_config
from train import select_dataset, select_model

def test(config):
    datamodule = select_dataset(config, 'test')
    model = select_model(config, datamodule)

    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    trainer = pl.Trainer(default_root_dir=config.models_save_path, weights_save_path=config.models_save_path, logger=logger)
    trainer.test(model=model, datamodule=datamodule)

if __name__ == '__main__':
    config = get_config()
    test(config)
