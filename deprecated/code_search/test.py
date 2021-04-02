import sys
import torch
import transformers 
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import BASE_DIR, get_config
from train import select_dataset, select_model

def test(config):
    datamodule = select_dataset(config, 'test')
    model = select_model(config, datamodule.tokenizer)

    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    ckpt_callback = ModelCheckpoint(monitor='val_rouge2_fmeasure', dirpath=config.models_save_path, save_top_k=3)

    trainer = pl.Trainer(logger=logger, resume_from_checkpoint=config.resume_ckpt, callbacks=[ckpt_callback],
                tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus,
                default_root_dir=config.models_save_path, weights_save_path=config.models_save_path)
    
    trainer.test(model=model, datamodule=datamodule)

def try_eg(config):
    sys.path.append(BASE_DIR)
    from dataset_scripts import get_tokenizer

    tokenizer = get_tokenizer(config)
    model = select_model(config, tokenizer)
    model.eval()

    code1 = input("Enter code 1: ")
    code2 = input("Enter code 2: ")
    query = input("Enter query: ")
    acc = model([query], [code1, code2]) 
    print("Labels: [0, 1]")
    print("Accuracy: ", acc)


if __name__ == '__main__':
    config = get_config()
    # test(config)
    try_eg(config)
