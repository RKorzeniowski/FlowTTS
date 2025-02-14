import os
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_utils import TextMelLoader, TextMelCollate, CustomPhonemeMelLoader, CustomAlignedPhonemeMelLoader
import models
import commons
import utils
from text.symbols import symbols


class FlowGenerator_DDI(models.FlowGenerator):
  """A helper for Data-dependent Initialization"""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for f in self.decoder.flows:
      if getattr(f, "set_ddi", False):
        f.set_ddi(True)


def main():
  hps = utils.get_hparams()
  logger = utils.get_logger(hps.model_dir)
  logger.info(hps)
  utils.check_git_hash(hps.model_dir)
  n_gpu_bs = hps.train.batch_size

  torch.manual_seed(hps.train.seed)

  if hps.train.loader_type == "lj":
    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
  else:
    if hps.train.loader_type == "custom":
        ds = CustomPhonemeMelLoader
    elif hps.train.loader_type == "custom_align":
        ds = CustomAlignedPhonemeMelLoader
    else:
        raise ValueError(f"dataset type {hps.train.loader_type} not supported")
    train_dataset = ds(
        data_paths=hps.data.train_files,
        phonemes_folder=hps.data.phonemes_folder,
        mel_folder=hps.data.mel_folder,
        suffix=hps.data.suffix,
        spec_transform=hps.data.get("spec_transform"),
    )

  collate_fn = TextMelCollate(2)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)

  generator = FlowGenerator_DDI(
      utils.get_vocab_size(hps.data.vocab_type, hps.data.vocab_file),
      out_channels=hps.data.n_mel_channels,
      external_alignment=hps.train.get("external_alignment"),
      **hps.model
  ).cuda()

  optimizer_g = commons.Adam(
      generator.parameters(), n_gpu_bs=n_gpu_bs, scheduler=hps.train.scheduler, dim_model=hps.model.hidden_channels,
      warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
   
  generator.train()
  for batch_idx, data in enumerate(train_loader):
    x, x_lengths = data[0].cuda(), data[1].cuda()
    y, y_lengths = data[2].cuda(), data[3].cuda()
    align = None
    if hps.train.loader_type == "custom_align":
        align = data[4].cuda(), data[5].cuda()

    _ = generator(x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, gen=False, external_alignment=align)
    break

  utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, 0, os.path.join(hps.model_dir, "ddi_G.pth"))

                            
if __name__ == "__main__":
  main()
