import torch
from utils import training_utils
from configs import new_custom_multiclass_config
from pipeline import logger as training_logger
from model import losses
from data_processing import datasets
from model import models

config = new_custom_multiclass_config
hparams = config.hparams
config = config.config
rank = 0

databunch = datasets.Databunch.from_config(config=config, hparams=hparams, rank=rank, bs=hparams.batch_size)
phon_alignment_predictor = models.PhonemeAlignmentPredictor(
                vocab_size=databunch.get_vocab_size(),
                txt_embed_dim=hparams.txt_embed_dim,
                lstm_hidden=hparams.enc_lstm_hidden,
                kernel_size=hparams.lenpred_conv1d_kernel_size,
                padding=hparams.lenpred_conv1d_padding
).cuda()
alignment_criterion = losses.PhonemeAlignLoss()
optimizer = torch.optim.Adam(
                phon_alignment_predictor.parameters(),
                lr=hparams.lr,
                betas=hparams.betas,
                weight_decay=hparams.weight_decay,
            )
logger = training_logger.Logger.from_config(ds_sizes=databunch.get_datasets_sizes(), config=config)


def predict_alignment(text, text_len):
    text_mask = training_utils.get_mask_from_lens(text_len)
    pred_alignment = phon_alignment_predictor(text, text_mask)
    return pred_alignment


def run_training():
    training_utils.set_seed(hparams.SEED)
    for epoch in range(1, hparams.epochs):
        run_epoch()


def unpack_batch(batch):
    data = {"alignment": None}
    text, text_len, spec, spec_len, align_padded, align_lengths, cls, cls_peak = batch
    data["cls_type"] = cls.to(rank)
    data["cls_peak"] = cls_peak.to(rank)
    data["alignment"] = align_padded.to(rank)
    data.update({
        "text": text.to(rank),
        "text_len": text_len.to(rank),
    })
    return spec, spec_len, data


def run_epoch(mode="train"):
    for step, batch in enumerate(databunch[mode]):
        optimization_logs = dict()
        spec, spec_len, data = unpack_batch(batch)
        pred_alignment = predict_alignment(data["text"], data["text_len"])
        alignment_loss = alignment_criterion(
                    pred_alignment=pred_alignment,
                    target_alignment=data["alignment"],
                )
        alignment_loss += alignment_loss * hparams.alignment_loss_coeff
        optimization_logs["alignment_loss"] = alignment_loss

        optimizer.zero_grad()
        alignment_loss.backward()
        optimizer.step()

        logger.update_step()
        logger.log(
            step=step,
            data=data,
            log_det=0,
            optimization_logs=optimization_logs,
        )

run_training()
