import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pipeline import pipeline
from configs import new_custom_multiclass_config


def main():
    config = new_custom_multiclass_config

    assert torch.cuda.is_available(), "CPU training is not supported."

    if config.hparams.distributed:
        n_gpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '80000'

        mp.spawn(train_distributed, nprocs=n_gpus, args=(n_gpus, config.config, config.hparams,))
    else:
        train(config.config, config.hparams, rank=0, n_gpus=1)


def train_distributed(rank, n_gpus, config, hparams):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)
    train(config=config, rank=rank, n_gpus=n_gpus, hparams=hparams)


def train(config, hparams, rank, n_gpus):
    if config.train_only_attn:
        training_pipeline = pipeline.VAETrainPipeline.get_attention_training(config=config, hparams=hparams, rank=rank, n_gpus=n_gpus)
    elif config.expectation_maximization_pipeline:
        training_pipeline = pipeline.EMVAETrainPipeline.from_config(config=config, hparams=hparams, rank=rank, n_gpus=n_gpus)
    elif config.multiclass:
        training_pipeline = pipeline.VAETrainPipeline.from_config(config=config, hparams=hparams, rank=rank, n_gpus=n_gpus)
    else:
        training_pipeline = pipeline.BasicTrainPipeline.from_config(config=config, hparams=hparams, rank=rank, n_gpus=n_gpus)
    training_pipeline.run_training()


if __name__ == "__main__":
    main()
