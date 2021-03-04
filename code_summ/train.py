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
    train_len = len(datamodule.train_dataloader())
    model = select_model(config, datamodule.tokenizer, train_len)
    
    logger = TensorBoardLogger(save_dir=config.tensorboard_path, name=config.exp_name)
    #ckpt_callback = ModelCheckpoint(monitor='val_bleu_score', dirpath=config.models_save_path, save_top_k=3)

    trainer = pl.Trainer(logger=logger, resume_from_checkpoint=config.resume_ckpt, num_sanity_val_steps=0, #callbacks=[ckpt_callback],
                tpu_cores=config.tpu_cores, gpus=config.gpus, auto_select_gpus=config.auto_select_gpus, precision=config.precision)
    
    trainer.fit(model, datamodule=datamodule)

def select_model(config, tokenizer=None, train_len=0):
    if config.model == 'bart':
        from models.bart import BartCode
        model = BartCode(config, tokenizer=tokenizer)
    elif config.model == 'mbart':
        from models.mbart import MBartCode
        model = MBartCode(config, tokenizer=tokenizer)
    elif config.model == 'p_codebert':
        from  models.p_codebert import PretrainedCodeBERT
        model = PretrainedCodeBERT(config, train_len, tokenizer=tokenizer)
    else:
        raise NotImplementedError
    
    if config.resume_ckpt is not None:
        checkpoint = torch.load(config.resume_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

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