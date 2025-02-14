from abc import abstractmethod
import pickle

import torch
import torch.nn as nn

from utils import training_utils
from model import flow_layers
from model import layers


class AbsFlowTTS(nn.Module):
    def __init__(
            self, txt_enc, decoder, alignment_smoothing, max_spec_seq_len,
            n_mel_channels, model_elements, alignment, pred_phoneme_alignment, n_sqz, *args, **kwargs
    ):
        super().__init__()
        self.text_encoder = txt_enc
        self.decoder = decoder
        self.alignment_smoothing = alignment_smoothing
        self.max_spec_seq_len = max_spec_seq_len
        self.n_mel_channels = n_mel_channels
        self.alignment = alignment
        self.pred_phoneme_alignment = pred_phoneme_alignment
        self.n_sqz = n_sqz
        for name, element in model_elements.items():
            self.add_module(name, element)

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def inference(self, **kwargs):
        pass

    @abstractmethod
    def sample_latent_vector(self, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def get_model_elements(cls, hparmas, class_count):
        pass

    def get_context(self, alignment, text, text_encodings, text_mask, spec_len, inference=False):
        pred_alignment = None
        if not self.alignment:
            positional_encoding = self.pos_enc(lens=spec_len)
            context_attn, pos_attn_weights = self.pos_attn(
                query=positional_encoding,
                key=text_encodings,
                value=text_encodings,
                key_padding_mask=~text_mask.squeeze(1),
            )
        else:
            if self.pred_phoneme_alignment:
                pred_alignment = self.phon_alignment_predictor(text, text_mask)
                if inference:
                    rounded_pred_alignment = pred_alignment.round()
                    alignment, align_mel_lengths = self.format_alignment(external_alignment=rounded_pred_alignment, max_align_lengths=rounded_pred_alignment.shape[1])
                    alignment = alignment.to(text.device)
            if not (inference and self.pred_phoneme_alignment):
                alignment = alignment.squeeze(1).transpose(1, 2)
            context_attn = self.upsample_alignment(alignment, text_encodings)
            spec_len = alignment.sum(dim=1).sum(dim=1).int()
        if self.alignment_smoothing:
            context_attn = self.alignment_smoother(context_attn)
        return context_attn, pred_alignment, spec_len

    def upsample_alignment(self, alignment, text_encodings):
        context_attn = torch.matmul(alignment, text_encodings.permute(1, 0, 2))
        context_attn = context_attn.permute(1, 0, 2)
        return context_attn

    def format_alignment(self, external_alignment, max_align_lengths):
        bs = external_alignment.shape[0]
        align_mel_lengths = external_alignment.sum(dim=1)
        max_mel_length, max_mel_idx = align_mel_lengths.max(0)
        if max_mel_length % self.n_sqz:
            align_mel_lengths[max_mel_idx] += self.n_sqz - max_mel_length % self.n_sqz
            max_mel_length = align_mel_lengths[max_mel_idx]

        formatted_alignment = torch.zeros((bs, int(max_mel_length.item()), max_align_lengths))
        # "needs more efficient implementation"
        for bs_idx in range(bs):
            sample_alignment = external_alignment[bs_idx]
            cum_sum = torch.cumsum(sample_alignment, dim=0)
            phonem_align_range = sum(sample_alignment)

            phonem_idx = 0
            for i in range(int(phonem_align_range.item())):
                while cum_sum[phonem_idx] <= i:
                    phonem_idx += 1
                formatted_alignment[bs_idx, i, phonem_idx] = 1

        return formatted_alignment, align_mel_lengths

    def predict_length(self, text_encodings, len_scale):
        spec_len = self.len_predictor(text_encodings) * len_scale
        spec_len = torch.clamp_max(torch.floor(spec_len.type(torch.FloatTensor)), max=self.max_spec_seq_len)
        return spec_len

    @classmethod
    def get_alignment_elements(self, hparams, alignment, bs, vocab_size, pred_phoneme_alignment):
        model_elements = {}
        if not alignment:
            model_elements["pos_attn"] = nn.MultiheadAttention(
                embed_dim=hparams.attn_embed_dim,
                num_heads=hparams.num_heads,
            )
            model_elements["len_predictor"] = layers.LengthPredictor(
                embed_dim=hparams.enc_lstm_hidden,
                kernel_size=hparams.lenpred_conv1d_kernel_size,
                padding=hparams.lenpred_conv1d_padding,
                dropout=hparams.dropout_lenpred,
            )
            model_elements["pos_enc"] = layers.PositionalEncoder(
                embedding_dim=hparams.enc_lstm_hidden,
                bs=bs,
                max_len=hparams.max_spec_seq_len,
            )
        if hparams.alignment_smoothing:
            model_elements["alignment_smoother"] = AlignmentSmoother(hparams.enc_lstm_hidden)

        if pred_phoneme_alignment:
            model_elements["phon_alignment_predictor"] = PhonemeAlignmentPredictor(
                vocab_size=vocab_size,
                txt_embed_dim=hparams.txt_embed_dim,
                lstm_hidden=hparams.enc_lstm_hidden,
                kernel_size=hparams.lenpred_conv1d_kernel_size,
                padding=hparams.lenpred_conv1d_padding,
                alignment_cls_embedding=hparams.alignment_cls_embedding,
            )
        return model_elements

    @classmethod
    def from_hparams(
            cls, hparams, vocab_size, alignment, bs, initialized,
            pred_phoneme_alignment, class_count, centroid_path, *args, **kwargs):
        decoder = Decoder(
            channel_dim=hparams.n_mel_channels,
            hidden_channels=hparams.dec_hidden_channels,
            num_blocks=hparams.dec_num_blocks,
            context_dim=hparams.enc_lstm_hidden,
            n_sqz=hparams.n_sqz,
            num_layers=hparams.num_dec_block_layers,
            initialized=initialized,
            flow_channel_drop_interval=hparams.flow_channel_drop_interval,
            flow_channel_drop_count=hparams.flow_channel_drop_count,
            act_normalization=hparams.act_norm,
            shared_dec_kernel_size=hparams.shared_dec_kernel_size,
            shared_dec_padding=hparams.shared_dec_padding,
            n_groups=hparams.n_groups,
            max_spec_seq_len=hparams.max_spec_seq_len,
            multiscale_arch=hparams.multiscale_arch,
            kernel_size=hparams.coupling_kernel_size,
            padding=hparams.coupling_padding,
            dropout=hparams.coupling_dropout,
        )
        txt_enc = TextEncoder(
            vocab_size=vocab_size,
            txt_embed_dim=hparams.txt_embed_dim,
            num_layers=hparams.num_enc_layers,
            lstm_hidden=hparams.enc_lstm_hidden,
            kernel_size=hparams.enc_conv1d_kernel_size,
            padding=hparams.enc_conv1d_padding,
            dropout=hparams.enc_dropout,
            bidir_enc=hparams.bidir_enc,
        )

        model_elements = cls.get_model_elements(
            hparams=hparams,
            class_count=class_count,
        )

        alignment_elements = cls.get_alignment_elements(
            hparams=hparams,
            alignment=alignment,
            bs=bs,
            vocab_size=vocab_size,
            pred_phoneme_alignment=pred_phoneme_alignment,
        )

        model_elements = {**model_elements, **alignment_elements}

        if centroid_path is not None:
            with open(centroid_path, 'rb') as f:
                centroid_values = pickle.load(f)
        else:
            centroid_values = None

        return cls(
            decoder=decoder,
            txt_enc=txt_enc,
            alignment_smoothing=hparams.alignment_smoothing,
            max_spec_seq_len=hparams.max_spec_seq_len,
            n_mel_channels=hparams.n_mel_channels,
            model_elements=model_elements,
            latent_prior=hparams.latent_prior,
            cls_embed_dim=hparams.cls_embed_dim,
            vae_downsample=hparams.vae_downsample,
            alignment=alignment,
            pred_phoneme_alignment=pred_phoneme_alignment,
            baseline_embedding=hparams.baseline_embedding,
            centroid_values=centroid_values,
            *args,
            **kwargs,
        )


class FlowTTS(AbsFlowTTS):
    def forward(self, text, text_len, spec, spec_len, alignment):
        aux_outputs = {}
        text_mask = training_utils.get_mask_from_lens(text_len)
        text_encodings = self.text_encoder(text, text_mask)

        if "len_predictor" in self._modules.keys():
            aux_outputs["pred_spec_length"] = self.len_predictor(text_encodings.detach())

        context_attn, pred_alignment, _ = self.get_context(
            alignment=alignment,
            text=text,
            text_encodings=text_encodings,
            text_mask=text_mask,
            spec_len=spec_len,
        )

        z, log_det = self.decoder.forward(
            x=spec,
            spec_len=spec_len,
            context_attn=context_attn,
        )

        spec_inv = self.decoder.reverse(z, spec_len=spec_len, context_attn=context_attn)
        aux_outputs["spec_inv"] = spec_inv
        aux_outputs["pred_alignment"] = pred_alignment
        return z, log_det, aux_outputs

    def inference(self, text, text_len, alignment, std=1., len_scale=1., external_align=False, **kwargs):
        with torch.no_grad():
            text_mask = training_utils.get_mask_from_lens(text_len)
            text_encodings = self.text_encoder(text, text_mask)

            spec_len = None if self.alignment else self.predict_length(text_encodings, len_scale)
            if external_align:
                context_attn = self.upsample_alignment(alignment.squeeze(1).transpose(1, 2), text_encodings)
                spec_len = alignment.sum(dim=1).sum(dim=1).int()
            else:
                context_attn, _, spec_len = self.get_context(
                    alignment=alignment,
                    text=text,
                    text_encodings=text_encodings,
                    text_mask=text_mask,
                    spec_len=spec_len,
                    inference=True,
                )
            z = self.sample_latent_vector(text, spec_len, std)

            spec = self.decoder.reverse(z, spec_len=spec_len, context_attn=context_attn)
            return spec

    def sample_latent_vector(self, text, spec_len, std):
        z = torch.randn((text.shape[0], self.n_mel_channels, spec_len.max().type(torch.LongTensor))) * std
        if text.device.type == 'cuda':
            z = z.cuda()
        return z

    @classmethod
    def get_model_elements(cls, hparams, class_count):
        elements = {}
        return elements


class VAEFlowTTS(AbsFlowTTS):
    def __init__(self, latent_prior, cls_embed_dim, vae_downsample, baseline_embedding, centroid_values, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_prior = latent_prior
        self.cls_embed_dim = cls_embed_dim
        self.vae_downsample = vae_downsample
        self.baseline_embedding = baseline_embedding
        self.centroid_inference = centroid_values is not None
        self.centroid_values = centroid_values

    def freeze_VAE(self, freeze):
        for name, param in self.named_parameters():
            if 'VAE' in name:
                param.requires_grad = not freeze

    def freeze_decoder(self, freeze):
        for name, param in self.named_parameters():
            if not 'VAE' in name:
                param.requires_grad = not freeze

    def forward(self, text, text_len, spec, spec_len, alignment, cls_type, cls_peak):
        aux_outputs = {}
        spec, spec_len = self.decoder.squeezer.trim_dims(spec, lens=spec_len)
        aux_outputs["spec"] = spec

        text_mask = training_utils.get_mask_from_lens(text_len)
        text_encodings = self.text_encoder(text, text_mask)

        if "len_predictor" in self._modules.keys():
            aux_outputs["pred_spec_length"] = self.len_predictor(text_encodings.detach())

        context, pred_alignment, _ = self.get_context(
            alignment=alignment,
            text=text,
            text_encodings=text_encodings,
            text_mask=text_mask,
            spec_len=spec_len,
        )

        context, mu, logs = self.combine_prior(context=context, spec=spec, cls_type=cls_type)

        aux_outputs["latent_target"] = {
            "mu": mu,
            "logs": logs,
            "target_mu": torch.zeros_like(mu) + cls_peak.unsqueeze(1),
            "target_logs": torch.log(torch.ones_like(logs))
        }

        z, log_det = self.decoder.forward(
            x=spec,
            spec_len=spec_len,
            context_attn=context,
        )

        spec_inv = self.decoder.reverse(z, spec_len=spec_len, context_attn=context)
        aux_outputs["spec_inv"] = spec_inv
        aux_outputs["pred_alignment"] = pred_alignment
        return z, log_det, aux_outputs

    def combine_prior(self, context, spec, cls_type):
        if self.latent_prior:
            if self.cls_embed_dim > 0:
                cls_embedding = self.class_embeddings(cls_type)
                context = self.context_combiner(attn=context, aux_context=cls_embedding)
            _, mu, logs = self.VAE(spec)
            if self.vae_downsample:
                mu, logs = self.vae_downsampler(m=mu, logs=logs)
        else:
            vae_context, mu, logs = self.VAE(spec)
            if self.baseline_embedding:
                vae_context = self.context_embedding(cls_type)
                mu = torch.zeros_like(mu)
                logs = torch.zeros_like(logs)
            context = self.context_combiner(attn=context, aux_context=vae_context)
        return context, mu, logs

    def inference(self, text, text_len, alignment, cls_type=None, cls_peak=None, std=1., len_scale=1., external_align=False, **kwargs):
        with torch.no_grad():
            text_mask = training_utils.get_mask_from_lens(text_len)
            text_encodings = self.text_encoder(text, text_mask)

            spec_len = None if self.alignment else self.predict_length(text_encodings, len_scale)
            if external_align:
                context = self.upsample_alignment(alignment.squeeze(1).transpose(1, 2), text_encodings)
                if self.alignment_smoothing:
                    context = self.alignment_smoother(context)
                spec_len = alignment.sum(dim=1).sum(dim=2).sum(dim=1).int()
            else:
                context, _, spec_len = self.get_context(
                    alignment=alignment,
                    text=text,
                    text_encodings=text_encodings,
                    text_mask=text_mask,
                    spec_len=spec_len,
                    inference=True,
                )

            z, context = self.combine_inference_prior(
                text=text,
                context=context,
                std=std,
                spec_len=spec_len,
                cls_peak=cls_peak,
                cls_type=cls_type,
            )
            spec = self.decoder.reverse(z, spec_len=spec_len, context_attn=context)
        return spec

    def combine_inference_prior(self, text, context, std, spec_len, cls_peak, cls_type):
        if self.centroid_inference:
            # assume bs = 1 during inference
            mu = self.centroid_values[cls_type.item()]["mu"]
            mu = torch.from_numpy(mu).to(text.device)
            dist_std = torch.ones_like(cls_peak).repeat(1, self.n_mel_channels) * std
        else:
            mu, dist_std = cls_peak, torch.ones_like(cls_peak) * std

        sampled_prior = self.sample_latent_vae_vector(
            spec_len=spec_len, mu=mu, std=dist_std, expand=not self.centroid_inference
        )

        if self.latent_prior:
            if self.cls_embed_dim > 0:
                cls_embedding = self.class_embeddings(cls_type)
                context = self.context_combiner(attn=context, aux_context=cls_embedding)
            z = sampled_prior
        else:
            vae_context = sampled_prior[:, :, 0]
            if self.baseline_embedding:
                vae_context = self.context_embedding(cls_type)
            context = self.context_combiner(attn=context, aux_context=vae_context)
            z = self.sample_latent_vector(text, spec_len, std)
        return z, context

    def sample_latent_vector(self, text, spec_len, std):
        z = torch.randn((text.shape[0], self.n_mel_channels, spec_len.max().type(torch.LongTensor))) * std
        if text.device.type == 'cuda':
            z = z.cuda()
        return z

    def sample_latent_vae_vector(self, spec_len, mu, std, expand=True):
        if expand:
            mu = mu.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        else:
            mu = torch.unsqueeze(mu, 2)
            std = torch.unsqueeze(std, 2)

        z = torch.randn((mu.shape[0], self.n_mel_channels, spec_len.max().type(torch.LongTensor)))
        if mu.device.type == 'cuda':
            z = z.cuda()
        z = z * std + mu
        return z

    @classmethod
    def get_model_elements(cls, hparams, class_count):
        elements = dict()
        elements["VAE"] = VAE(
            channels=hparams.n_mel_channels,
            hidden=hparams.vae_hidden,
            kernel=hparams.vae_kernel_size,
            padding=hparams.vae_padding,
        ).cuda()
        elements["context_combiner"] = ContextCombiner(
            txt_embed_dim=hparams.enc_lstm_hidden,
            vae_dim=0 if hparams.latent_prior else hparams.vae_hidden,
            cls_embedd_dim=hparams.cls_embed_dim if hparams.latent_prior else 0,
            bicontext_comb=hparams.bicontext_comb,
        ).cuda()
        if hparams.latent_prior:
            elements["vae_downsampler"] = VAEDownsampler(
                channels=hparams.n_mel_channels,
                kernel=hparams.vae_kernel_size,
                padding=hparams.vae_padding,
            ).cuda()
            if hparams.cls_embed_dim > 0:
                elements["class_embeddings"] = torch.nn.Embedding(
                     num_embeddings=class_count,
                     embedding_dim=hparams.cls_embed_dim,
                ).cuda()
        if hparams.baseline_embedding:
            elements["context_embedding"] = torch.nn.Embedding(
                     num_embeddings=class_count,
                     embedding_dim=hparams.vae_hidden,
                ).cuda()
        return elements


class ConditionalFlowTTS(VAEFlowTTS):
    def combine_prior(self, context, spec, cls_type):
        cls_embedding = self.class_embeddings(cls_type)
        context = self.context_combiner(attn=context, aux_context=cls_embedding)
        return context, None, None


class Decoder(nn.Module):
    def __init__(
            self,
            channel_dim,
            hidden_channels,
            num_blocks,
            context_dim,
            n_sqz,
            num_layers,
            initialized,
            flow_channel_drop_interval,
            flow_channel_drop_count,
            max_spec_seq_len,
            act_normalization,
            shared_dec_kernel_size,
            shared_dec_padding,
            multiscale_arch,
            n_groups,
            kernel_size,
            padding,
            dropout,
    ):
        super().__init__()
        self.squeezer = flow_layers.Squeezer(n_sqz, max_spec_seq_len)

        splitter = flow_layers.Splitter(
            multiscale_arch=multiscale_arch,
            channel_dim=channel_dim * n_sqz,
            num_blocks=num_blocks,
            drop_interval=flow_channel_drop_interval,
            drop_count=flow_channel_drop_count
        )

        self.channel_dims = splitter.calculate_splits()

        self.flow_steps = nn.ModuleList([
            flow_layers.FlowStep(
                channels=dim,
                hidden_channels=hidden_channels,
                context_dim=context_dim,
                n_sqz=n_sqz,
                num_layers=num_layers,
                act_normalization=act_normalization,
                shared_dec_kernel_size=shared_dec_kernel_size,
                shared_dec_padding=shared_dec_padding,
                n_groups=n_groups,
                kernel_size=kernel_size,
                padding=padding,
                dropout=dropout,
                initialized=initialized,
            )
            for dim in self.channel_dims[:-1]
        ])

        self.final_flow_step = flow_layers.FlowStep(
            channels=channel_dim * n_sqz,
            context_dim=context_dim,
            n_sqz=n_sqz,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            act_normalization=act_normalization,
            shared_dec_kernel_size=shared_dec_kernel_size,
            shared_dec_padding=shared_dec_padding,
            n_groups=n_groups,
            kernel_size=kernel_size,
            padding=padding,
            dropout=dropout,
            initialized=initialized,
        )

    def get_log_det(self):
        return {"log_det_w": 0, "log_det_std": 0, "log_det_n": 0, "total_log_det": 0}

    def update_log_det(self, log_det, w, std, n):
        log_det["log_det_w"] += w
        log_det["log_det_std"] += std
        log_det["log_det_n"] += n
        log_det["total_log_det"] += w + std + n
        return log_det

    def forward(self, x, spec_len, context_attn):
        x, spec_len = self.squeezer.trim_dims(x, lens=spec_len)
        x, x_mask = self.squeezer.squeeze(x, lens=spec_len)

        context_attn = context_attn.permute(1, 2, 0)

        context_attn, spec_len = self.squeezer.trim_dims(context_attn, lens=spec_len)
        context_attn, x_mask = self.squeezer.squeeze(context_attn, lens=spec_len)

        pre_output, log_det = [], self.get_log_det()
        full_mask = x_mask
        for flow_step, split_point in zip(self.flow_steps, self.channel_dims):
            x, out = training_utils.split(x, split_point=split_point)
            x_mask, _ = training_utils.split(x_mask, split_point=split_point)
            x, p_log_det_w, p_log_det_std, p_log_det_n = flow_step(
                x=x, mask=x_mask, context_attn=context_attn)
            self.update_log_det(log_det=log_det, w=p_log_det_w, std=p_log_det_std, n=p_log_det_n)
            pre_output.insert(0, out)
        pre_output.insert(0, x)
        x = torch.cat(pre_output, dim=1)
        x, f_log_det_w, f_log_det_std, f_log_det_n = self.final_flow_step(
            x=x,
            mask=full_mask,
            context_attn=context_attn,
        )
        output, mask = self.squeezer.unsqueeze(x, mask=x_mask)
        self.update_log_det(log_det=log_det, w=f_log_det_w, std=f_log_det_std, n=f_log_det_n)
        return output, log_det

    def reverse(self, z, spec_len, context_attn):
        z, spec_len = self.squeezer.trim_dims(z, lens=spec_len)
        z, z_mask = self.squeezer.squeeze(z, lens=spec_len)
        context_attn = context_attn.permute(1, 2, 0)
        context_attn, spec_len = self.squeezer.trim_dims(context_attn, lens=spec_len)
        context_attn, z_mask = self.squeezer.squeeze(context_attn, lens=spec_len)
        z = self.final_flow_step.reverse(z=z, mask=z_mask, context_attn=context_attn)
        for flow_step, split_point in reversed(list(zip(self.flow_steps, self.channel_dims))):
            z, out = training_utils.split(z, split_point=split_point)
            z_mask, _ = training_utils.split(z_mask, split_point=split_point)
            z = flow_step.reverse(z=z, mask=z_mask, context_attn=context_attn)
            z = torch.cat([z, out], dim=1)
        x, mask = self.squeezer.unsqueeze(z, mask=z_mask)
        return x


class TextEncoder(nn.Module):
    def __init__(
            self,
            num_layers,
            vocab_size,
            txt_embed_dim,
            lstm_hidden,
            kernel_size,
            padding,
            bidir_enc,
            dropout,
    ):
        super().__init__()

        self.symbol_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=txt_embed_dim,
        )

        self.encoder_blocks = nn.ModuleList([
            layers.TextEncoderBlock(
                in_channels=txt_embed_dim,
                out_channels=txt_embed_dim,
                kernel_size=kernel_size,
                padding=padding,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(
            input_size=txt_embed_dim,
            hidden_size=lstm_hidden,
            bidirectional=bidir_enc,
        )

    def forward(self, x, mask):
        z = self.symbol_embedding(x).permute((0, 2, 1))
        mask = mask.type(z.type()).cuda()
        for enc_block in self.encoder_blocks:
            z = enc_block(z) * mask
        z, _ = self.lstm(z.permute(2, 0, 1))
        z = (z[:, :, :self.lstm_hidden] + z[:, :, self.lstm_hidden:]).permute(1, 2, 0)
        z = z.permute(2, 0, 1)
        return z


class VAE(nn.Module):
    def __init__(self, channels, hidden, kernel, padding):
        super().__init__()
        self.enc_conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel,
            padding=padding,
            stride=2
        )
        self.enc_bn1 = nn.BatchNorm1d(channels)
        self.enc_tanh1 = nn.Tanh()
        self.enc_conv2_mean = nn.Conv1d(
            in_channels=channels,
            out_channels=hidden,
            kernel_size=kernel,
            padding=padding,
            stride=4
        )
        self.enc_bn2_mean = nn.BatchNorm1d(hidden)
        self.enc_tanh2_mean = nn.Tanh()

        self.enc_conv2_std = nn.Conv1d(
            in_channels=channels,
            out_channels=hidden,
            kernel_size=kernel,
            padding=padding,
            stride=8,
        )
        self.enc_bn2_std = nn.BatchNorm1d(hidden)
        self.enc_tanh2_std = nn.Tanh()

        self.lstm_m = nn.LSTM(input_size=hidden, hidden_size=hidden)
        self.lstm_std = nn.LSTM(input_size=hidden, hidden_size=hidden)

    def encode(self, x):
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_tanh1(x)

        # x  # [bs, ch, phonem_len] # teraz mu logs dla kazdego phonemu

        mu = self.enc_conv2_mean(x)
        mu = self.enc_bn2_mean(mu)
        mu = self.enc_tanh2_mean(mu)

        logs = self.enc_conv2_std(x)
        logs = self.enc_bn2_std(logs)
        logs = self.enc_tanh2_std(logs)

        mu = mu.permute(2, 0, 1)
        logs = logs.permute(2, 0, 1)
        mu, _ = self.lstm_m(mu)
        logs, _ = self.lstm_std(logs)
        mu = mu.permute(1, 2, 0)[:, :, -1]
        logs = logs.permute(1, 2, 0)[:, :, -1]
        return mu, logs

        # jak sampluje to N(0, 1) nie ma sensu bo wtedy nie rozrozniam fonemow
        # centroid dla kazdego fonema oddzielnie

    def reparameterize(self, mu, logs):
        std = torch.exp(logs / 2)
        eps = torch.randn_like(std).cuda()
        return mu + eps * std

    def forward(self, x):
        mu, logs = self.encode(x)
        z = self.reparameterize(mu, logs)
        return z, mu, logs


class ContextCombiner(nn.Module):
    def __init__(self, txt_embed_dim, vae_dim, bicontext_comb, cls_embedd_dim):
        super().__init__()
        input_size = txt_embed_dim + vae_dim + cls_embedd_dim
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=txt_embed_dim, bidirectional=bicontext_comb)
        self.lstm_hidden = txt_embed_dim
        self.bicontext_comb = bicontext_comb

    def forward(self, attn, aux_context):
        aux = torch.stack(attn.shape[0] * [aux_context])
        z = torch.cat([aux, attn], dim=2)
        z, _ = self.lstm(z)
        if self.bicontext_comb:
            z = (z[:, :, :self.lstm_hidden] + z[:, :, self.lstm_hidden:])
        return z


class VAEDownsampler(nn.Module):
    def __init__(self, channels, kernel, padding):
        super().__init__()
        self.mean_conv = nn.Linear(in_features=channels, out_features=1)
        self.std_conv = nn.Linear(in_features=channels, out_features=1)

    def forward(self, m, logs):
        d_m = self.mean_conv(m)
        d_logs = self.std_conv(logs)
        return d_m, d_logs


class AlignmentSmoother(nn.Module):
    def __init__(self, spec_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=spec_dim, hidden_size=spec_dim, bidirectional=True)
        self.lstm_hidden = spec_dim

    def forward(self, x):
        z, _ = self.lstm(x)
        z = (z[:, :, :self.lstm_hidden] + z[:, :, self.lstm_hidden:])
        return z


class PhonemeAlignmentPredictor(nn.Module):
    def __init__(
            self,
            vocab_size,
            txt_embed_dim,
            lstm_hidden,
            kernel_size,
            padding,
            alignment_cls_embedding,
            ):
        super().__init__()
        self.symbol_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=txt_embed_dim,
        )
        if alignment_cls_embedding:
            self.cls_embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=txt_embed_dim // 4,
            )
            conv_in_dim = txt_embed_dim + txt_embed_dim // 4
        else:
            conv_in_dim = txt_embed_dim

        self.conv1d = nn.Conv1d(
            in_channels=conv_in_dim,
            out_channels=txt_embed_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.batch_norm = nn.BatchNorm1d(txt_embed_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=txt_embed_dim, hidden_size=lstm_hidden, bidirectional=True)
        self.lstm_hidden = lstm_hidden
        self.linear = nn.Linear(in_features=self.lstm_hidden, out_features=1)
        self.alignment_cls_embedding = alignment_cls_embedding

    def forward(self, x, mask):
        z = self.symbol_embedding(x).permute((0, 2, 1))
        if self.alignment_cls_embedding:
            cls_embed = self.cls_embedding(x).permute((0, 2, 1))
            z = torch.cat([z, cls_embed], dim=1)
        mask = mask.type(z.type()).cuda()
        z = self.conv1d(z) * mask
        z = self.relu(self.batch_norm(z))
        z, _ = self.lstm(z.permute(2, 0, 1))
        z = (z[:, :, :self.lstm_hidden] + z[:, :, self.lstm_hidden:]).permute(1, 2, 0)
        z = self.relu(z) * mask
        z = self.linear(z.permute(0, 2, 1)) * mask.permute(0, 2, 1)
        z = self.relu(z.permute(0, 2, 1))
        return z.squeeze(1)
