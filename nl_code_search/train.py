import torch
import transformers 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# Use this file to select model based on the config, load the pretrained model etc. and call pl.Trainer()

from config import get_config

def select_model(config):
    model, datamodule = None, None
    if config.model == 'pretrained_codebert':
        from pretrained_codebert.models import PretrainedCodeBERT
        from pretrained_codebert.dataset import PretrainedCodeBERTDataModule

        tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/codebert-base')
        datamodule = PretrainedCodeBERTDataModule(config, tokenizer)
        total_steps = len(datamodule.train_dataloader(batch_size=config.batch_size)) // config.n_epochs

        model = PretrainedCodeBERT(config, total_steps, tokenizer=tokenizer)

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
