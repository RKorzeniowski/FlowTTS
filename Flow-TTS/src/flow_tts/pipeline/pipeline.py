from abc import abstractmethod
import itertools

import torch
from pathlib import Path
import shutil
import os
from os import listdir
import numpy as np
import subprocess
from torch.nn.parallel import DataParallel as DP

from data_processing import datasets
from model import losses
from model import models
from pipeline import logger as training_logger
from pipeline import lr_schedulers
from utils import training_utils


class TrainPipeline:
    def __init__(
        self,
        model,
        optimizer,
        databunch,
        flow_loss_fn,
        len_loss_fn,
        logger,
        checkpoint_path,
        multiclass,
        hparams,
        config,
        rank,
    ):
        self.model = model
        self.optimizer = optimizer
        self.databunch = databunch
        self.flow_loss_fn = flow_loss_fn
        self.len_loss_fn = len_loss_fn
        self.logger = logger
        self.l1loss = torch.nn.L1Loss(reduction='sum')
        self.checkpoint_path = checkpoint_path
        self.multiclass = multiclass
        self.vae_loss = losses.VAELoss(hparams.loss_type) if multiclass else None
        self.lr_scheduler = lr_schedulers.lr_schedulers[hparams.lr_scheduler](
            optimizer=self.optimizer,
            lr=hparams.lr,
            steps_per_epoch=len(databunch.datasets["train"]),
            epochs=hparams.epochs,
        ) if hparams.lr_scheduler else None
        self.alignment_criterion = losses.PhonemeAlignLoss() if config.pred_phoneme_alignment else None
        self.hparams = hparams
        self.config = config
        self.rank = rank

    def run_training(self):
        training_utils.set_seed(self.hparams.SEED)
        for epoch in range(1, self.hparams.epochs):
            self.model.train()
            self.run_epoch(epoch, mode='train')
            if self.rank == 0:
                with torch.no_grad():
                    self.logger.set_mode('eval')
                    self.model.eval()
                    self.run_epoch(epoch, mode='eval')
                    self.logger.set_mode('train')
                    self.save_checkpoint()
                    self.logger.update_epoch()
                    self.generate_samples(epoch)

    @abstractmethod
    def run_epoch(self, epoch, mode='train'):
        pass

    @abstractmethod
    def run_inference_pipeline(self, epoch):
        pass

    def include_alignment_loss(self, flow_loss, optimization_logs, aux_outputs, spec_len, alignment):
        total_loss = flow_loss
        if "pred_spec_length" in aux_outputs.keys():
            pred_spec_len = aux_outputs["pred_spec_length"]
            float_pec_len = spec_len.type(torch.FloatTensor).to(self.rank)
            len_loss = self.len_loss_fn(torch.log1p(pred_spec_len), torch.log1p(float_pec_len))
            total_loss += len_loss
            optimization_logs["len_loss"] = len_loss
            optimization_logs["pred_spec_len"] = pred_spec_len

        if self.config.pred_phoneme_alignment:
            alignment_loss = self.alignment_criterion(
                pred_alignment=aux_outputs["pred_alignment"],
                target_alignment=alignment,
            )
            total_loss += alignment_loss * self.hparams.alignment_loss_coeff
            optimization_logs["alignment_loss"] = alignment_loss

        return total_loss, optimization_logs

    def generate_samples(self, epoch):
        if epoch % self.config.generate_samples_epoch_interval == 0:
            suffix = f"_epoch_{epoch}"
            self.run_inference_pipeline(suffix)
            self.run_audio_pred(
                synth_command=self.config.synth_command,
                data_path=self.config.data_path,
                vocoder_path=self.config.vocoder_path,
                folder_name_filter=self.config.folder_name_filter,
                target_folder_prefix=self.config.target_folder_prefix,
            )

    def run_audio_pred(self, synth_command, data_path, vocoder_path, folder_name_filter, target_folder_prefix):
        data_path = Path(data_path)
        cwd = os.getcwd()
        os.chdir(vocoder_path)
        for folder_name in listdir(data_path):
            if folder_name_filter in folder_name:
                source_path = data_path / folder_name
                target_path = data_path / (target_folder_prefix + folder_name.replace(folder_name_filter, ""))
                target_path.mkdir(exist_ok=True)
                command = synth_command.format(source_path=source_path, target_path=target_path)
                subprocess.run(command, shell=True)

        os.chdir(cwd)

    def log(self, step, data, log_det, optimization_logs):
        if self.rank == 0 and self.logger.is_step_logged(step):
            self.logger.update_step()
            model = self.model.module if self.hparams.distributed else self.model
            pred_align_spec = model.inference(external_align=True, **data)
            pred_align_spec = torch.clamp(pred_align_spec, min=0)
            l1_aligned_loss = self.calculate_l1_loss(spec=data["spec"], spec_inv=pred_align_spec,
                                                     spec_len=data["spec_len"])
            optimization_logs["l1_aligned_loss"] = l1_aligned_loss
            optimization_logs["pred_align_spec"] = pred_align_spec

            print("log")
            try:
                pred_spec = model.inference(**data)
                pred_spec = torch.clamp(pred_spec, min=0)
                l1_loss = self.calculate_l1_loss(spec=data["spec"], spec_inv=pred_spec, spec_len=data["spec_len"])
                optimization_logs["l1_loss"] = l1_loss
                optimization_logs["pred_spec"] = pred_spec
            except Exception as E:
                print(E)

            self.logger.log(
                step=step,
                data=data,
                log_det=log_det,
                optimization_logs=optimization_logs,
            )

    def update_model(self, mode, total_loss):
        if mode == 'train':
            self.optimizer.zero_grad()
            total_loss.backward()

            training_utils.clip_grad_value(
                self.model.parameters(),
                clip_value=self.hparams.grad_clip_value,
            )
            self.optimizer.step()

    def unpack_batch(self, batch):
        data = {"alignment": None}
        if self.multiclass:
            text, text_len, spec, spec_len, align_padded, align_lengths, cls, cls_peak = batch
            data["cls_type"] = cls.to(self.rank)
            data["cls_peak"] = cls_peak.to(self.rank)
        else:
            text, text_len, spec, spec_len, align_padded, align_lengths = batch
        data["alignment"] = align_padded.to(self.rank)
        spec = spec.to(self.rank)
        spec_len = spec_len.to(self.rank)
        data.update({
            "text": text.to(self.rank),
            "text_len": text_len.to(self.rank),
            "spec": spec,
            "spec_len": spec_len
        })
        return spec, spec_len, data

    def calculate_l1_loss(self, spec, spec_inv, spec_len):
        spec, spec_len = self.model.decoder.squeezer.trim_dims(x=spec, lens=spec_len)
        spec_inv, spec_len = self.model.decoder.squeezer.trim_dims(x=spec_inv, lens=spec_len)
        mask = training_utils.get_mask_from_lens(spec_len)
        shorter_len = min(spec.shape[2], spec_inv.shape[2])
        spec = spec[..., :shorter_len]
        spec_inv = spec_inv[..., :shorter_len]
        mask = mask[..., :shorter_len]
        spec = spec * mask
        spec_inv = spec_inv * mask
        l1_loss = self.l1loss(spec, spec_inv)
        l1_loss /= torch.sum(spec_len) * self.hparams.n_mel_channels
        return l1_loss

    def save_checkpoint(self):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'hparams': self.hparams,
            'config': self.config,
            'step': self.logger.global_step,
            'epoch': self.logger.epoch,
            },
            self.checkpoint_path
        )

    @classmethod
    @abstractmethod
    def get_model(cls, config, hparams, vocab_size, bs, rank, initialized=False):
        pass

    @classmethod
    def get_optimizer(cls, hparams, model, parameters):
        if hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=hparams.lr,
                betas=hparams.betas,
                weight_decay=hparams.weight_decay,
            )
        elif hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=hparams.lr,
                momentum=hparams.momentum,
                weight_decay=hparams.weight_decay,
            )
        else:
            raise ValueError()
        return optimizer

    @classmethod
    def from_checkpoint(cls, checkpoint_path, rank, n_gpus):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint['hparams']
        config = checkpoint['config']
        bs = hparams.batch_size // n_gpus if hparams.distributed else hparams.batch_size

        #### remove
        hparams.baseline_embedding = False
        hparams.alignment_cls_embedding = False
        config.centroid_path = "/home/ec2-user/renard/Flow-TTS/src/flow_tts/centroids_VAE_all_dcVAE_new12block.pkl"
        ####

        new_checkpoint_path = Path(config.model_dir) / config.checkpoint_path
        databunch = datasets.Databunch.from_config(hparams=hparams, config=config, rank=rank, bs=bs)
        vocab_size = databunch.get_vocab_size()
        ds_sizes = databunch.get_datasets_sizes()
        if rank == 0:
            logger = training_logger.Logger.from_config(
                ds_sizes=ds_sizes, config=config, epoch=checkpoint['epoch'], global_step=0)#checkpoint['global_step'])
        else:
            logger = None
        model = cls.get_model(config, hparams, vocab_size, bs, rank, initialized=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer = cls.get_optimizer(hparams, model, model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        return cls(
            model=model,
            optimizer=optimizer,
            databunch=databunch,
            flow_loss_fn=losses.FlowLoss(hparams.loss_norm_mask, hparams.n_sqz, hparams.n_mel_channels).to(rank),
            len_loss_fn=torch.nn.MSELoss().to(rank),
            logger=logger,
            checkpoint_path=new_checkpoint_path,
            multiclass=config.multiclass,
            hparams=hparams,
            config=config,
            rank=rank,
        )

    @classmethod
    def get_attention_training(cls, config, hparams, rank, n_gpus):
        checkpoint = torch.load(config.checkpoint_path)
        chkp_config = checkpoint['config']
        bs = hparams.batch_size // n_gpus if hparams.distributed else hparams.batch_size

        databunch = datasets.Databunch.from_config(hparams=hparams, config=chkp_config, rank=rank, bs=bs)
        vocab_size = databunch.get_vocab_size()
        ds_sizes = databunch.get_datasets_sizes()
        logger = training_logger.Logger.from_config(
            ds_sizes=ds_sizes, config=config, epoch=0, global_step=0)

        # chkp_config["alignment"] = False
        # chkp_config["train_only_attn"] = True
        # chkp_config["dec_num_blocks"] = 12

        model = cls.get_model(chkp_config, hparams, vocab_size, bs, rank, initialized=True)
        try:
            model.load_state_dict(checkpoint['model'], strict=False)
        except Exception as e:
            print(e)

        for p in model.parameters():
            p.requires_grad = False

        # align_elements = [model.pos_attn, model.len_predictor, model.pos_enc, model.alignment_smoother] # normally without align smoother
        # for m in align_elements:
        #     for p in m.parameters():
        #         p.requires_grad = True
        #
        # params = itertools.chain(*[x.parameters() for x in align_elements])
        optimizer = cls.get_optimizer(hparams, model, model.paramters())
        return cls(
            model=model,
            optimizer=optimizer,
            databunch=databunch,
            flow_loss_fn=losses.FlowLoss(hparams.loss_norm_mask, hparams.n_sqz, hparams.n_mel_channels).to(rank),
            len_loss_fn=torch.nn.MSELoss().to(rank),
            logger=logger,
            checkpoint_path=Path(config.model_dir) / "checkpoint_model.pth",
            multiclass=config.multiclass,
            hparams=hparams,
            config=config,
            rank=rank,
        )

    @classmethod
    def from_config(cls, config, hparams, rank, n_gpus):
        bs = hparams.batch_size // n_gpus if hparams.distributed else hparams.batch_size
        checkpoint_path = Path(config.model_dir) / config.checkpoint_path
        databunch = datasets.Databunch.from_config(config=config, hparams=hparams, rank=rank, bs=bs)
        vocab_size = databunch.get_vocab_size()
        ds_sizes = databunch.get_datasets_sizes()
        if rank == 0:
            logger = training_logger.Logger.from_config(ds_sizes=ds_sizes, config=config)
        else:
            logger = None
        model = cls.get_model(config, hparams, vocab_size, bs, rank)
        optimizer = cls.get_optimizer(hparams, model, model.parameters())
        return cls(
            model=model,
            optimizer=optimizer,
            databunch=databunch,
            flow_loss_fn=losses.FlowLoss(hparams.loss_norm_mask, hparams.n_sqz, hparams.n_mel_channels).to(rank),
            len_loss_fn=torch.nn.MSELoss().to(rank),
            logger=logger,
            checkpoint_path=checkpoint_path,
            multiclass=config.multiclass,
            hparams=hparams,
            config=config,
            rank=rank,
        )


class BasicTrainPipeline(TrainPipeline):
    def run_epoch(self, epoch, mode='train'):
        for step, batch in enumerate(self.databunch[mode]):
            spec, spec_len, data = self.unpack_batch(batch)
            z, log_det, aux_outputs = self.model.forward(
                **data,
            )
            flow_loss, pdf = self.flow_loss_fn(z=z, spec_len=spec_len, log_det=log_det['total_log_det'])
            optimization_logs = dict(flow_loss=flow_loss, loss=flow_loss, pdf=pdf, z=z, aux_outputs=aux_outputs)

            total_loss, optimization_logs = self.include_alignment_loss(
                flow_loss=flow_loss,
                optimization_logs=optimization_logs,
                aux_outputs=aux_outputs,
                spec_len=spec_len,
                alignment=data["alignment"] if self.config.pred_phoneme_alignment else None,
            )

            self.log(
                step=step,
                data=data,
                log_det=log_det,
                optimization_logs=optimization_logs,
            )
            self.update_model(mode, total_loss)

    def run_inference_pipeline(self, epoch):
        inf_pipeline = InferencePipeline.from_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            samples_count=self.config.samples_count,
            data_path=self.config.data_path,
            txt_file=self.config.txt_file,
            modes=self.config.modes,
            stds=None,
            overwrite_style=True,
            sample_classes=None,
            styles=None,
            rank=0,
        )
        inf_pipeline.inference_known_align()

    @classmethod
    def get_model(cls, config, hparams, vocab_size, bs, rank, initialized=False):
        model = models.FlowTTS.from_hparams(
            hparams=hparams,
            vocab_size=vocab_size,
            alignment=config.alignment,
            bs=bs,
            initialized=initialized,
            pred_phoneme_alignment=config.pred_phoneme_alignment,
            n_sqz=hparams.n_sqz,
            class_count=config.class_count,
        ).to(rank)
        if hparams.distributed:
            model = DP(model, device_ids=[rank])
        return model


class ConditionalTrainPipeline(TrainPipeline):
    def run_epoch(self, epoch, mode="train"):
        for step, batch in enumerate(self.databunch[mode]):
            spec, spec_len, data = self.unpack_batch(batch)
            z, log_det, aux_outputs = self.model.forward(
                **data,
            )

            flow_loss, pdf = self.flow_loss_fn(z=z, spec_len=spec_len, log_det=log_det["total_log_det"])

            optimization_logs = dict(
                flow_loss=flow_loss, loss=flow_loss, pdf=pdf, z=z,
                aux_outputs=aux_outputs,
            )

            total_loss = flow_loss

            total_loss, optimization_logs = self.include_alignment_loss(
                flow_loss=total_loss,
                optimization_logs=optimization_logs,
                aux_outputs=aux_outputs,
                spec_len=spec_len,
                alignment=data["alignment"] if self.config.pred_phoneme_alignment else None,
            )

            if flow_loss.isnan():
                raise ValueError()

            self.log(
                step=step,
                data=data,
                log_det=log_det,
                optimization_logs=optimization_logs,
            )
            self.update_model(mode, total_loss)
            if self.lr_scheduler is not None: self.lr_scheduler.step()

    def run_inference_pipeline(self, epoch):
        pass

    @classmethod
    def get_model(cls, config, hparams, vocab_size, bs, rank, initialized=False):
        model = models.ConditionalFlowTTS.from_hparams(
            hparams=hparams,
            vocab_size=vocab_size,
            alignment=config.alignment,
            bs=bs,
            initialized=initialized,
            pred_phoneme_alignment=config.pred_phoneme_alignment,
            n_sqz=hparams.n_sqz,
            class_count=config.class_count,
        ).to(rank)
        if hparams.distributed:
            model = DP(model, device_ids=[rank])
        return model


class VAETrainPipeline(TrainPipeline):
    def run_epoch(self, epoch, mode="train"):
        for step, batch in enumerate(self.databunch[mode]):
            spec, spec_len, data = self.unpack_batch(batch)
            z, log_det, aux_outputs = self.model.forward(
                **data,
            )

            vae_params = aux_outputs["latent_target"]
            vae_loss = self.vae_loss(**vae_params)

            if self.hparams.latent_prior:
                flow_loss, pdf = self.flow_loss_fn(
                    z=z,
                    spec_len=spec_len,
                    log_det=log_det["total_log_det"],
                    target=vae_params,
                )
            else:
                flow_loss, pdf = self.flow_loss_fn(z=z, spec_len=spec_len, log_det=log_det["total_log_det"])

            optimization_logs = dict(
                flow_loss=flow_loss, loss=flow_loss, pdf=pdf, z=z,
                aux_outputs=aux_outputs, vae_loss=vae_loss,
                vae_mu=vae_params["mu"], vae_logs=vae_params["logs"],
            )

            total_loss = flow_loss

            if self.is_include_vae_loss():
                flow_loss += vae_loss

            total_loss, optimization_logs = self.include_alignment_loss(
                flow_loss=total_loss,
                optimization_logs=optimization_logs,
                aux_outputs=aux_outputs,
                spec_len=spec_len,
                alignment=data["alignment"] if self.config.pred_phoneme_alignment else None,
            )

            if flow_loss.isnan():
                raise ValueError()

            self.log(
                step=step,
                data=data,
                log_det=log_det,
                optimization_logs=optimization_logs,
            )
            self.update_model(mode, total_loss)
            if self.lr_scheduler is not None: self.lr_scheduler.step()

    def run_inference_pipeline(self, suffix):
        inf_pipeline = VAEInferencePipeline.from_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            samples_count=self.config.samples_count,
            data_path=self.config.data_path,
            txt_file=self.config.txt_file,
            modes=self.config.modes,
            stds=self.config.stds,
            overwrite_style=self.config.overwrite_style,
            sample_classes=self.config.sample_classes,
            styles=self.config.styles,
            suffix=suffix,
            rank=0,
        )
        inf_pipeline.inference_known_align()

    def is_include_vae_loss(self):
        global_step = self.logger.global_step
        return global_step > self.hparams.VAE_headstart and global_step % self.hparams.VAE_update_interval == 0

    @classmethod
    def get_model(cls, config, hparams, vocab_size, bs, rank, initialized=False):
        model = models.VAEFlowTTS.from_hparams(
            hparams=hparams,
            vocab_size=vocab_size,
            alignment=config.alignment,
            bs=bs,
            initialized=initialized,
            pred_phoneme_alignment=config.pred_phoneme_alignment,
            n_sqz=hparams.n_sqz,
            class_count=config.class_count,
            centroid_path=config.centroid_path,
        ).to(rank)
        if hparams.distributed:
            model = DP(model, device_ids=[rank])
        return model


class EMVAETrainPipeline(VAETrainPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = "VAE"

    def run_epoch(self, epoch, mode="train"):
        for step, batch in enumerate(self.databunch[mode]):
            self.switch_freeze(step)
            spec, spec_len, data = self.unpack_batch(batch)
            z, log_det, aux_outputs = self.model.forward(
                **data,
            )

            vae_params = aux_outputs["latent_target"]
            optimization_logs = dict(
                z=z, aux_outputs=aux_outputs,
                vae_mu=vae_params["mu"], vae_logs=vae_params["logs"],
            )

            if self.phase == "decoder":
                loss, pdf = self.flow_loss_fn(
                    z=z,
                    spec_len=spec_len,
                    log_det=log_det["total_log_det"],
                    target=vae_params,
                )
                optimization_logs["flow_loss"], optimization_logs["pdf"] = loss, pdf
            else:
                loss = self.vae_loss(**vae_params)
                optimization_logs["vae_loss"] = loss
                step -= 1

            total_loss, optimization_logs = self.include_alignment_loss(
                flow_loss=loss,
                optimization_logs=optimization_logs,
                aux_outputs=aux_outputs,
                spec_len=spec_len,
                alignment=data["alignment"] if self.config.pred_phoneme_alignment else None,
            )

            self.log(
                step=step,
                data=data,
                log_det=log_det,
                optimization_logs=optimization_logs,
            )
            self.update_model(mode, total_loss)

    def switch_freeze(self, step):
        if step % self.config.switch_dec_interval == 0:
            self.phase = "VAE"
            self.model.freeze_VAE(False)
            self.model.freeze_decoder(True)
        else:
            self.phase = "decoder"
            self.model.freeze_VAE(True)
            self.model.freeze_decoder(False)

    def run_inference_pipeline(self, suffix):
        pass

    def update_model(self, mode, total_loss):
        if mode == 'train':
            self.optimizer[self.phase].zero_grad()
            total_loss.backward()

            training_utils.clip_grad_value(
                self.model.parameters(),
                clip_value=self.hparams.grad_clip_value,
            )
            self.optimizer[self.phase].step()

    @classmethod
    def get_optimizer(cls, hparams, model, parameters):
        params_VAE, params_decoder = [], []
        for name, param in model.named_parameters():
            if 'VAE' in name:
                params_VAE.append(param)
            else:
                params_decoder.append(param)
        VAE_optimizer = torch.optim.Adam(
            params_VAE,
            lr=hparams.EM_VAE_lr,
            betas=hparams.betas,
            weight_decay=hparams.weight_decay,
        )
        decoder_optimizer = torch.optim.Adam(
            params_VAE,
            lr=hparams.lr,
            betas=hparams.betas,
            weight_decay=hparams.weight_decay,
        )
        return {"VAE": VAE_optimizer, "decoder": decoder_optimizer}

    def save_checkpoint(self):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': (self.optimizer["VAE"].state_dict(), self.optimizer["decoder"].state_dict()),
            'hparams': self.hparams,
            'config': self.config,
            'step': self.logger.global_step,
            'epoch': self.logger.epoch,
            },
            self.checkpoint_path
        )


class AbsInferencePipeline:
    def __init__(
            self, model, databunch, samples_count, data_path, txt_file,
            rank, modes, stds, multiclass, suffix, hparams, *args, **kwargs
    ):
        self.model = model
        self.databunch = databunch
        self.samples_count = samples_count
        self.modes = modes
        self.stds = stds
        self.txt_file = txt_file
        self.data_path = Path(data_path)
        self.target_spec_folder = self.data_path / "target_specs_{mode}"
        self.pred_spec_folder = self.data_path / "pred_specs_{mode}_{std}{suffix}"
        self.pred_img_folder = self.data_path / "img_spec_{mode}_{std}{suffix}"
        self.multiclass = multiclass
        self.suffix = suffix
        self.hparams = hparams
        self.rank = rank

    def inference(self, text):
        text_len = text.shape
        pred_spec = self.model.inference(text=text, text_len=text_len)
        return pred_spec

    def inference_known_len(self):
        pass

    def get_blank_name_from_path(self, path):
        return Path(path).stem

    def copy_targets(self, src, mode):
        target_spec_folder = Path(str(self.target_spec_folder).format(mode=mode))
        target_spec_folder.mkdir(parents=True, exist_ok=True)
        target = target_spec_folder / Path(src).name
        shutil.copy(src, target)

    def save_spec_img(self, y, sample_name, std, mode):
        pred_img_folder = Path(str(self.pred_img_folder).format(std=std, mode=mode, suffix=self.suffix))
        pred_img_folder.mkdir(parents=True, exist_ok=True)
        save_img_path = pred_img_folder / sample_name
        training_utils.plot_spectrogram_to_numpy(y, save_path=save_img_path)

    def save_spec_npz(self, y, sample_name, std, mode):
        pred_spec_folder = Path(str(self.pred_spec_folder).format(std=std, mode=mode, suffix=self.suffix))
        pred_spec_folder.mkdir(parents=True, exist_ok=True)
        save_npz_path = pred_spec_folder / sample_name
        npz_gen_tst = np.moveaxis(y, 0, -1)
        with open(save_npz_path.with_suffix(".npz"), "wb") as f:
            np.savez(f, spectrogram=npz_gen_tst)

    def save_txt(self, txt, sample_name, idx, mode):
        save_path = self.data_path / self.txt_file.format(mode=mode)
        with open(save_path, "a") as f:
            record = f"{idx} {sample_name} {txt}\n"
            f.write(record)

    @classmethod
    @abstractmethod
    def get_model(cls, config, hparams, vocab_size, bs, rank, initialized):
        pass

    @classmethod
    def from_checkpoint(
            cls, checkpoint_path, samples_count, data_path,
            txt_file, rank, modes, stds, suffix="", *args, **kwargs
            ):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint['hparams']
        config = checkpoint['config']
        bs = hparams.batch_size

        #### remove
        hparams.baseline_embedding = False
        hparams.alignment_cls_embedding = False
        config.centroid_path = "/home/ec2-user/renard/Flow-TTS/src/flow_tts/centroids_VAE_all_dcVAE_new12block.pkl"
        ####

        databunch = datasets.Databunch.from_config(config=config, hparams=hparams, inference=True, rank=rank, bs=bs)
        vocab_size = databunch.get_vocab_size()
        model = cls.get_model(config=config, hparams=hparams, vocab_size=vocab_size, bs=bs, rank=rank, initialized=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        model = model.cuda()
        model = model.eval()
        return cls(
            model=model,
            databunch=databunch,
            samples_count=samples_count,
            data_path=data_path,
            txt_file=txt_file,
            rank=rank,
            modes=modes,
            stds=stds,
            alignment=config.alignment,
            multiclass=config.multiclass,
            suffix=suffix,
            hparams=hparams,
            **kwargs
        )


class InferencePipeline(AbsInferencePipeline):
    def inference_known_align(self):
        for mode in self.modes:
            for i in range(self.samples_count):
                spec_path, phonem_path, alignment, prompt = self.databunch[mode].dataset.sample_paths[i]

                phonemes = self.databunch[mode].dataset.get_phonemes(path=phonem_path).unsqueeze(0).to(self.rank)
                phonemes_len = torch.LongTensor([phonemes.shape[1]]).to(self.rank)

                alignment = [torch.LongTensor(alignment)]
                alignment = torch.nn.utils.rnn.pad_sequence(alignment, batch_first=True)
                align_lengths = torch.LongTensor([x.shape[0] for x in alignment])
                alignment, align_lengths = datasets.unpack_alignment(alignment, align_lengths, div=self.hparams.n_sqz)
                alignment = alignment.to(self.rank)

                pred_spec = self.model.inference(
                    text=phonemes,
                    text_len=phonemes_len,
                    alignment=alignment,
                )
                np_spec = pred_spec[0].cpu().data.numpy()

                sample_name = self.get_blank_name_from_path(spec_path)
                self.copy_targets(src=spec_path, mode=mode)
                self.save_txt(txt=prompt, sample_name=sample_name, idx=i, mode=mode)
                self.save_spec_npz(y=np_spec, sample_name=sample_name, std=str(1.0), mode=mode)
                self.save_spec_img(y=np_spec, sample_name=sample_name, std=str(1.0), mode=mode)

    @classmethod
    def get_model(cls, config, hparams, vocab_size, bs, rank, initialized=False):
        model = models.FlowTTS.from_hparams(
            hparams=hparams,
            vocab_size=vocab_size,
            alignment=config.alignment,
            bs=bs,
            initialized=initialized,
            pred_phoneme_alignment=config.pred_phoneme_alignment,
            n_sqz=hparams.n_sqz,
            class_count=config.class_count,
        )
        if hparams.distributed:
            model = DP(model, device_ids=[rank])
        return model


class VAEInferencePipeline(AbsInferencePipeline):
    def __init__(self, overwrite_style, sample_classes, styles=((0, 0.0),), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overwrite_style = overwrite_style
        self.sample_classes = sample_classes
        self.styles = styles

    def inference_known_align(self):
        for mode in self.modes:
            for std in self.stds:
                for new_class_idx, mean in self.styles:
                    for target_cls in self.sample_classes:
                        samples = self.databunch[mode].dataset.get_class_subset(target_cls)
                        samples_count = min(self.samples_count, len(samples))
                        for i in range(samples_count):
                            melspec_path, phonemes_path, alignment, prompt, class_idx, class_peak_value = samples[i]

                            phonemes = self.databunch[mode].dataset.get_phonemes(path=phonemes_path).unsqueeze(0).to(
                                self.rank)
                            phonemes_len = torch.LongTensor([phonemes.shape[1]]).to(self.rank)

                            alignment = [torch.LongTensor(alignment)]
                            alignment = torch.nn.utils.rnn.pad_sequence(alignment, batch_first=True)
                            align_lengths = torch.LongTensor([x.shape[0] for x in alignment])
                            alignment, align_lengths = datasets.unpack_alignment(alignment, align_lengths, div=self.hparams.n_sqz)
                            alignment = alignment.to(self.rank)

                            if self.overwrite_style:
                                class_idx = new_class_idx
                                class_peak_value = mean

                            class_idx = torch.LongTensor([class_idx]).to(self.rank)
                            cls_peak = torch.LongTensor([class_peak_value]).to(self.rank)

                            pred_spec = self.model.inference(
                                text=phonemes,
                                text_len=phonemes_len,
                                alignment=alignment,
                                std=std,
                                cls_type=class_idx,
                                cls_peak=cls_peak,
                            )
                            np_spec = pred_spec[0].cpu().data.numpy()

                            sample_name = self.get_blank_name_from_path(melspec_path)
                            if self.overwrite_style:
                                sample_name = "mean_" + str(mean).replace(".", "_") + "cls_idx" + str(
                                    new_class_idx) + sample_name
                            self.copy_targets(src=melspec_path, mode=mode)
                            self.save_txt(txt=prompt, sample_name=sample_name, idx=i, mode=mode)
                            self.save_spec_npz(y=np_spec, sample_name=sample_name, std=std, mode=mode)
                            self.save_spec_img(y=np_spec, sample_name=sample_name, std=std, mode=mode)

    @classmethod
    def get_model(cls, config, hparams, vocab_size, bs, rank, initialized):
        model = models.VAEFlowTTS.from_hparams(
            hparams=hparams,
            vocab_size=vocab_size,
            alignment=config.alignment,
            bs=bs,
            initialized=True,
            pred_phoneme_alignment=config.pred_phoneme_alignment,
            n_sqz=hparams.n_sqz,
            class_count=config.class_count,
            centroid_path=config.centroid_path,
        )
        if hparams.distributed:
            model = DP(model, device_ids=[rank])
        return model
