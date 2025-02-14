import os
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from data_utils import TextMelLoader, TextMelCollate, CustomPhonemeMelLoader, CustomAlignedPhonemeMelLoader
import models
import commons
import utils


global_step = 0

def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80000'

  hps = utils.get_hparams()
  mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, hps,))


def train_and_eval(rank, n_gpus, hps):
  n_gpu_bs = hps.train.batch_size * n_gpus
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

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

  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextMelCollate(2)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn, sampler=train_sampler)

  if rank == 0:
    if hps.train.loader_type == "lj":
      val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
    else:
      if hps.train.loader_type == "custom":
        ds = CustomPhonemeMelLoader
      elif hps.train.loader_type == "custom_align":
        ds = CustomAlignedPhonemeMelLoader
      else:
        raise ValueError(f"dataset type {hps.train.loader_type} not supported")
      val_dataset = ds(
        data_paths=hps.data.eval_files,
        phonemes_folder=hps.data.phonemes_folder,
        mel_folder=hps.data.mel_folder,
        suffix=hps.data.suffix,
        spec_transform=hps.data.get("spec_transform"),
      )

    val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=True, collate_fn=collate_fn)


  n_vocab = utils.get_vocab_size(hps.data.vocab_type, hps.data.vocab_file)
  generator = models.FlowGenerator(
      n_vocab=n_vocab,
      out_channels=hps.data.n_mel_channels,
      external_alignment=hps.train.get("external_alignment"),
      **hps.model
  ).cuda(rank)

  optimizer_g = commons.Adam(
    generator.parameters(), n_gpu_bs=n_gpu_bs, scheduler=hps.train.scheduler, dim_model=hps.model.hidden_channels,
    warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
  if hps.train.fp16_run:
    generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")
  generator = DDP(generator)
  epoch_str = 1
  global_step = 0
  try:
    _, _, _, epoch_str = utils.load_checkpoint(checkpoint_path=utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                                               model=generator, optimizer=optimizer_g, n_gpu_bs=n_gpu_bs)
    epoch_str += 1
    global_step = optimizer_g.step_num
    print(f"LOADED MODEL ON EPOCH {epoch_str}")
  except Exception as exp:
    print(f"FAILED TO LOAD MODEL. Exception {exp}")
    if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
      _ = utils.load_checkpoint(checkpoint_path=os.path.join(hps.model_dir, "ddi_G.pth"),
                        model=generator, optimizer=optimizer_g, n_gpu_bs=n_gpu_bs)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer)
      evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval)
      utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_latest.pth".format(epoch)))
    else:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)


def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
  train_loader.sampler.set_epoch(epoch)
  global global_step

  generator.train()

  for batch_idx, data in enumerate(train_loader):
    x, x_lengths = data[0].cuda(rank, non_blocking=True), data[1].cuda(rank, non_blocking=True)
    y, y_lengths = data[2].cuda(rank, non_blocking=True), data[3].cuda(rank, non_blocking=True)
    align = None
    
    optimizer_g.zero_grad()

    (z, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs = generator(x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, gen=False, external_alignment=align)
    const = 0.5 * math.log(2 * math.pi)
    seq_len_norm = (torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
    logp_z = torch.sum(y_logs) + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m)**2)
    logp_x = (logp_z - torch.sum(logdet))

    l_mle = const + logp_x / seq_len_norm
    l_length = torch.sum((logw - logw_)**2) / torch.sum(x_lengths)
    loss_gs = [l_mle, l_length]
    loss_g = sum(loss_gs)

    if hps.train.fp16_run:
      with amp.scale_loss(loss_g, optimizer_g._optim) as scaled_loss:
        scaled_loss.backward()
      grad_norm = commons.clip_grad_value_(amp.master_params(optimizer_g._optim), 5)
    else:
      loss_g.backward()
      grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
    optimizer_g.step()

    if rank == 0:
      if batch_idx % hps.train.log_interval == 0:
        print()

        eval_align = (align[0][:1], align[1].max().unsqueeze(dim=0)) if align else None
        (y_gen, *_), *_ = generator.module(x=x[:1], x_lengths=x_lengths[:1], gen=True, external_alignment=eval_align)

        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x), len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss_g.item()))
        logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.get_lr()])

        scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()),
            "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()),
            "attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy()),
            },
          scalars=scalar_dict)
    global_step += 1

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))


def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
  if rank == 0:
    global global_step
    generator.eval()
    losses_tot = []
    with torch.no_grad():
      for batch_idx, data in enumerate(val_loader):
        x, x_lengths = data[0].cuda(rank, non_blocking=True), data[1].cuda(rank, non_blocking=True)
        y, y_lengths = data[2].cuda(rank, non_blocking=True), data[3].cuda(rank, non_blocking=True)
        align = None
        
        (z, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs = generator(x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, gen=False, external_alignment=align)
        l_mle = 0.5 * math.log(2 * math.pi) + (torch.sum(y_logs) + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m)**2) - torch.sum(logdet)) / (torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels)
        l_length = torch.sum((logw - logw_)**2) / torch.sum(x_lengths)
        loss_gs = [l_mle, l_length]
        loss_g = sum(loss_gs)

        if batch_idx == 0:
          losses_tot = loss_gs
        else:
          losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

        if batch_idx % hps.train.log_interval == 0:
          logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(x), len(val_loader.dataset),
            100. * batch_idx / len(val_loader),
            loss_g.item()))
          logger.info([x.item() for x in loss_gs])


    losses_tot = [x/len(val_loader) for x in losses_tot]
    loss_tot = sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
      writer=writer_eval,
      global_step=global_step,
      scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))


if __name__ == "__main__":
  main()
