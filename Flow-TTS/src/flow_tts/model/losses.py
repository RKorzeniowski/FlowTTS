import torch


class FlowLoss(torch.nn.Module):
    def __init__(self, loss_norm_mask, n_sqz, n_mel_channels):
        super().__init__()
        self.loss_norm_mask = loss_norm_mask
        self.n_sqz = n_sqz
        self.n_mel_channels = n_mel_channels

    def forward(self, z, spec_len, log_det, target=None):

        sigma = 1.0
        if self.loss_norm_mask:
            norm = torch.sum(spec_len // self.n_sqz) * self.n_sqz * self.n_mel_channels
        else:
            norm = z.size(0) * z.size(1) * z.size(2)

        if target is not None:
            y_m, y_logs = target["mu"], target["logs"]
            if y_logs.shape[1] == self.n_mel_channels:
                y_logs = torch.stack(z.size(2) * [y_logs]).permute(1, 2, 0)
                y_m = y_m.unsqueeze(-1)
            elif y_logs.shape[1] == 1:
                y_logs = torch.ones_like(z) * y_logs.unsqueeze(-1)
                y_m = y_m.unsqueeze(-1)
            else:
                raise ValueError()

            logp_z = torch.sum(y_logs) + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2)
        else:
            logp_z = torch.sum(z * z) / (2 * sigma * sigma)

        loss = logp_z - log_det
        loss = loss / norm
        return loss, logp_z


class VAELoss(torch.nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, mu, logs, target_mu, target_logs):
        if self.loss_type == "basic":
            kl_div = - 0.5 * torch.mean(1 + logs - mu.pow(2) - logs.exp())
        elif self.loss_type == "prior":
            p = torch.distributions.normal.Normal(loc=mu, scale=logs.exp())
            q = torch.distributions.normal.Normal(loc=target_mu, scale=target_logs.exp())
            kl_div = torch.distributions.kl.kl_divergence(p, q).mean()
        else:
            raise ValueError()
        return kl_div


class PhonemeAlignLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, pred_alignment, target_alignment):
        target_alignment = target_alignment.squeeze().sum(dim=2)
        loss = self.criterion(pred_alignment, target_alignment)
        return loss
