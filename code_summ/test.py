import torch
import transformers 
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import get_config
from train import select_dataset, select_model

def test(config):
    datamodule = select_dataset(config, 'test')
    model = select_model(config, datamodule.tokenizer)

    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    ckpt_callback = ModelCheckpoint(monitor='val_bleu_score', dirpath=config.models_save_path, save_top_k=3)

    trainer = pl.Trainer(logger=logger, resume_from_checkpoint=config.resume_ckpt, callbacks=[ckpt_callback],
                tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus,
                default_root_dir=config.models_save_path, weights_save_path=config.models_save_path)
    
    trainer.test(model=model, datamodule=datamodule)

def try_eg(config):
    datamodule = select_dataset(config)
    model = select_model(config, datamodule.tokenizer)

    code = input("Enter input : ")
    print(model(code))

if __name__ == '__main__':
    config = get_config()
    #test(config)
    try_eg(config)