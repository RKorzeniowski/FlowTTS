import torch
import torch.nn as nn

from utils import training_utils


class Squeezer(nn.Module):
    def __init__(self, n_sqz, max_spec_len):
        super().__init__()
        self.n_sqz = n_sqz
        self.max_spec_len = max_spec_len

    def squeeze(self, x, lens=None):
        b, c, t = x.size()

        t = (t // self.n_sqz) * self.n_sqz
        x = x[:, :, :t]
        x_sqz = x.view(b, c, t // self.n_sqz, self.n_sqz)
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * self.n_sqz, t // self.n_sqz)

        if lens is not None:
            mask = training_utils.get_mask_from_lens(lens)
            mask = mask[:, :, self.n_sqz - 1::self.n_sqz]
        else:
            mask = torch.ones(b, 1, t // self.n_sqz).to(device=x.device, dtype=x.dtype)
        mask = mask.type(x.type())
        return x_sqz * mask, mask

    def unsqueeze(self, x, mask=None):
        b, c, t = x.size()

        x_unsqz = x.view(b, self.n_sqz, c // self.n_sqz, t)
        x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // self.n_sqz, t * self.n_sqz)

        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.n_sqz).view(b, 1, t * self.n_sqz)
        else:
            mask = torch.ones(b, 1, t * self.n_sqz).to(device=x.device, dtype=x.dtype)
        mask = mask.type(x.type())
        return x_unsqz * mask, mask

    def trim_dims(self, x, lens):
        max_length = lens.max()
        max_length = (max_length // self.n_sqz) * self.n_sqz
        x = x[:, :, :max_length.type(torch.LongTensor).cuda()]
        lens = (lens // self.n_sqz) * self.n_sqz
        return x, lens


class Splitter:
    def __init__(self, multiscale_arch, channel_dim, num_blocks, drop_interval, drop_count):
        self.multiscale_arch = multiscale_arch
        self.channel_count = channel_dim
        self.drop_interval = drop_interval
        self.channel_drop_count = drop_count
        self.num_blocks = num_blocks

    def calculate_splits(self):
        if self.multiscale_arch:
            channels = [self.channel_count - self.channel_drop_count * (i // self.drop_interval)
                        for i in range(self.num_blocks)]
        else:
            channels = [self.channel_count for _ in range(self.num_blocks)]
        return channels


class FlowStep(nn.Module):
    def __init__(
            self, channels, hidden_channels, context_dim, num_layers, n_sqz, act_normalization,
            shared_dec_kernel_size, shared_dec_padding, n_groups, kernel_size, padding, dropout, initialized
    ):
        super().__init__()
        self.act_normalization = act_normalization

        self.act_norm = ActNorm(channels, initialized=initialized) if act_normalization else None
        self.inv_conv = Invertible1x1Conv(channels=channels, n_split=n_groups)
        self.affine_coupling = AffineCouplingLayer(
            channels=channels,
            hidden_channels=hidden_channels,
            txt_enc_dim=context_dim,
            n_sqz=n_sqz,
            num_layers=num_layers,
            kernel_size=kernel_size,
            padding=padding,
            dropout=dropout,
        )
        context_input_dim = context_dim * n_sqz

        self.context_conv = WeigthNormConv1d(
            in_channels=context_input_dim,
            out_channels=2 * hidden_channels,
            kernel_size=shared_dec_kernel_size,
            padding=shared_dec_padding,
        )

    def forward(self, x, mask, context_attn):
        if self.act_normalization:
            x, log_det_n = self.act_norm(x=x, mask=mask)
        else:
            log_det_n = 0
        context_attn = self.context_conv(context_attn)
        norm_x, log_det_w = self.inv_conv.forward(x, mask=mask)
        z, log_det_std = self.affine_coupling.forward(norm_x, mask=mask, context_attn=context_attn)
        return z, log_det_w, log_det_std, log_det_n

    def reverse(self, z, mask, context_attn):
        context_attn = self.context_conv(context_attn)
        x_norm = self.affine_coupling.reverse(
            z,
            mask=mask,
            context_attn=context_attn,
        )
        x = self.inv_conv.reverse(x_norm, mask=mask)
        if self.act_normalization: x = self.act_norm.reverse(z=x, mask=mask)
        return x


class AffineCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, txt_enc_dim, num_layers, n_sqz, kernel_size, padding, dropout):
        super().__init__()
        self.coupling_block = CouplingBlock(
            channels=channels,
            hidden_channels=hidden_channels,
            txt_enc_dim=txt_enc_dim,
            n_sqz=n_sqz,
            num_layers=num_layers,
            kernel_size=kernel_size,
            padding=padding,
            dropout=dropout,
        )

    def forward(self, x, mask, context_attn):
        x1, x2 = training_utils.split(x, split_point=x.shape[1] // 2)
        mean, log_std = self.coupling_block(x2, mask, context_attn)
        std = torch.exp(log_std)
        z1 = (x1 * std + mean) * mask
        z2 = x2
        z = torch.cat([z1, z2], dim=1)
        log_det = torch.sum(log_std * mask)
        return z, log_det

    def reverse(self, z, mask, context_attn):
        z1, z2 = training_utils.split(z, split_point=z.shape[1] // 2)
        x2 = z2
        mean, log_std = self.coupling_block(x2, mask, context_attn)
        inv_std = torch.exp(-log_std)
        x1 = ((z1 - mean) * inv_std) * mask
        x = torch.cat([x1, x2], dim=1)
        return x


class CouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, txt_enc_dim, num_layers, n_sqz, kernel_size, padding, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.gtus, self.convs1d, self.res_skip_cons = \
            torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()

        double_hid_channels = 2 * hidden_channels
        for i in range(num_layers):
            skip_con_dim = double_hid_channels if i < num_layers - 1 else hidden_channels
            self.gtus.append(GatedTanhUnit(channels=double_hid_channels, txt_enc_dim=txt_enc_dim, n_sqz=n_sqz))
            self.convs1d.append(WeigthNormConv1d(
                in_channels=hidden_channels,
                out_channels=double_hid_channels,
                kernel_size=kernel_size,
                padding=padding,
            ))
            self.res_skip_cons.append(WeigthNormConv1d(hidden_channels, skip_con_dim, kernel_size=1))
        self.dropout = nn.Dropout(dropout)

        self.pre_conv1x1 = WeigthNormConv1d(
            in_channels=channels // 2,
            out_channels=hidden_channels,
            kernel_size=1,
        )
        self.post_conv1x1 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.post_conv1x1.weight.data.zero_()
        self.post_conv1x1.bias.data.zero_()

    def forward(self, x, mask, context_attn):
        x = self.pre_conv1x1(x) * mask

        out = torch.zeros_like(x)
        for i in range(self.num_layers):
            x_in = self.convs1d[i](x)
            x_in = self.dropout(x_in)
            acts = self.gtus[i](x_in, context_attn)
            res_skip_acts = self.res_skip_cons[i](acts)
            if self.is_last_layer(i):
                out = out + res_skip_acts
            else:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * mask
                out = out + res_skip_acts[:, self.hidden_channels:, :]
        out = self.post_conv1x1(out * mask)

        mean, log_std = training_utils.split(out, split_point=out.shape[1] // 2)
        return mean, log_std

    def is_last_layer(self, i):
        return i >= self.num_layers - 1


class GatedTanhUnit(nn.Module):
    def __init__(self, channels, txt_enc_dim, n_sqz):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        out_channels = channels // 2
        self.spec_conv1d = WeigthNormConv1d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.txt_conv1d = WeigthNormConv1d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x, context_attn):
        return self.sigmoid(self.spec_conv1d(x)) * self.tanh(self.txt_conv1d(context_attn))


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class Invertible1x1Conv(nn.Module):
    def __init__(self, channels, n_split):
        super().__init__()
        assert (n_split % 2 == 0)
        self.channels = channels
        self.n_split = n_split

        w_init = torch.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, mask):
        b, c, t = x.size()
        assert (c % self.n_split == 0)
        x_len = torch.sum(mask, [1, 2])

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

        weight = self.weight
        logdet = torch.sum(torch.logdet(self.weight) * (c / self.n_split) * x_len)

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = nn.functional.conv2d(x, weight)

        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * mask
        return z, logdet

    def reverse(self, x, mask):
        b, c, t = x.size()
        assert (c % self.n_split == 0)

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

        if hasattr(self, "weight_inv"):
            weight = self.weight_inv
        else:
            weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = nn.functional.conv2d(x, weight)

        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * mask
        return z


class ActNorm(nn.Module):
    def __init__(self, channels, initialized):
        super().__init__()
        self.channels = channels
        self.initialized = initialized

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, mask):
        x_len = torch.sum(mask, [1, 2])
        if not self.initialized:
            self.initialize(x, mask)
            self.initialized = True

        z = (self.bias + torch.exp(self.logs) * x) * mask
        logdet = torch.sum(torch.sum(self.logs) * x_len)

        return z, logdet

    def reverse(self, z, mask):
        assert self.initialized, "Layer has to be first initialized with data to make prediction"
        x = (z - self.bias) * torch.exp(-self.logs) * mask
        return x

    def set_initialized(self, initialized):
        self.initialized = initialized

    def initialize(self, x, mask):
        with torch.no_grad():
            denom = torch.sum(mask, [0, 2])
            m = torch.sum(x * mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * mask, [0, 2]) / denom
            v = m_sq - (m ** 2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class WeigthNormConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = torch.nn.Conv1d(*args, **kwargs)
        self.norm_conv = torch.nn.utils.weight_norm(conv)

    def forward(self, x):
        return self.norm_conv(x)
