from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from utils import training_utils


class Logger:
    def __init__(self, ds_sizes, logging_interval, model_dir, epoch, global_step, mode='train'):
        self.mode = mode
        self.logging_interval = logging_interval
        self.ds_sizes = ds_sizes
        self.writers = {name: SummaryWriter(log_dir=model_dir + "/" + name) for name in ds_sizes.keys()}
        self.epoch = epoch
        self.global_step = global_step

    def update_epoch(self, step=1):
        self.epoch += step

    def set_mode(self, mode):
        self.mode = mode

    def pdf(self, data, mean: float, variance: float):
        s1 = 1 / (np.sqrt(2 * np.pi * variance))
        s2 = np.exp(-(np.square(data - mean) / (2 * variance)))
        return s1 * s2

    def plot_pdf(self, pred, means, stds):
        X = pred.cpu().flatten().detach().numpy()[:300]

        min_arg = np.argmin(means)
        max_arg = np.argmax(means)
        min_std = 3 * stds[min_arg] if means[min_arg] > 0 else -3 * stds[min_arg]
        max_std = 3 * stds[max_arg] if means[max_arg] > 0 else -3 * stds[max_arg]

        bins = np.linspace(means[min_arg] + min_std, means[max_arg] * max_std, 100)
        fig, ax = plt.subplots()

        ax.scatter(X, [0.005] * len(X), s=1, alpha=0.5)

        for mean, std in zip(means, stds):
            ax.plot(bins, self.pdf(bins, mean, std), color='red')

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def update_step(self):
        self.global_step += self.logging_interval

    def log(self, step, data, log_det, optimization_logs):
        writer = self.writers[self.mode]
        if 'flow_loss' in optimization_logs:
            print(f"\nepoch: {self.epoch}, step: {step}/{self.ds_sizes[self.mode]}\n"
                  f"flow loss: {optimization_logs['flow_loss']}\n")
            writer.add_scalar("aa flow loss", optimization_logs["flow_loss"], global_step=self.global_step)
            writer.add_scalar("flow latent mean", optimization_logs["z"].mean(), global_step=self.global_step)
            writer.add_scalar("flow latent std", optimization_logs["z"].std(), global_step=self.global_step)
            writer.add_scalar("flow pdf", optimization_logs["pdf"], global_step=self.global_step)
            writer.add_scalar("flow log_det_inv_conv", log_det["log_det_w"], global_step=self.global_step)
            writer.add_scalar("flow log_det_coupling", log_det["log_det_std"], global_step=self.global_step)
            writer.add_scalar("flow log_det_act_norm", log_det["log_det_n"], global_step=self.global_step)
        if "vae_loss" in optimization_logs.keys():
            print(f'vae_loss: {optimization_logs["vae_loss"]}')
            writer.add_scalar("vae_loss", optimization_logs["vae_loss"], global_step=self.global_step)
            writer.add_scalar("vae_mu", optimization_logs["vae_mu"].mean(), global_step=self.global_step)
            writer.add_scalar("vae_std", optimization_logs["vae_logs"].exp().mean(), global_step=self.global_step)
        if "l1_loss" in optimization_logs.keys():
            writer.add_scalar("spec batch l1 loss", optimization_logs["l1_loss"], global_step=self.global_step)
            writer.add_scalar("spec aligned batch l1 loss", optimization_logs["l1_aligned_loss"], global_step=self.global_step)

        if step % 20 == 0 and "pred_align_spec" in optimization_logs.keys():
            pred_align_spec = optimization_logs["pred_align_spec"][:1].data.cpu().numpy()
            target_spec = data["spec"][:1].data.cpu().numpy()

            target_img = training_utils.plot_spectrogram_to_numpy(target_spec[0])
            pred_align_spec = training_utils.plot_spectrogram_to_numpy(pred_align_spec[0])

            writer.add_image("aa y_align_pred", np.moveaxis(pred_align_spec, [0, 1], [-2, -1]), self.global_step)
            writer.add_image("y_org", np.moveaxis(target_img, [0, 1], [-2, -1]), self.global_step)

            if step % 13 == 0:
                d = optimization_logs['aux_outputs']["spec_inv"][0].data.cpu().numpy()
                indentity_inv_img = training_utils.plot_spectrogram_to_numpy(d)
                writer.add_image("identity_inv", np.moveaxis(indentity_inv_img, [0, 1], [-2, -1]), self.global_step)

            if "l1_loss" in optimization_logs.keys():
                pred_spec = optimization_logs["pred_spec"][:1].data.cpu().numpy()
                pred_img = training_utils.plot_spectrogram_to_numpy(pred_spec[0])
                writer.add_image("aa y_pred", np.moveaxis(pred_img, [0, 1], [-2, -1]), self.global_step)

            if "z" in optimization_logs.keys():
                z_img = training_utils.plot_spectrogram_to_numpy(optimization_logs["z"][0].data.cpu().numpy())
                writer.add_image("z", np.moveaxis(z_img, [0, 1], [-2, -1]), self.global_step)

        if "len_loss" in optimization_logs.keys():
            writer.add_scalar("len_loss", optimization_logs["len_loss"], global_step=self.global_step)
        if "pred_spec_len" in optimization_logs.keys():
            optimization_logs["len_diff"] = abs(optimization_logs["pred_spec_len"] - data["spec_len"]).mean()
        if "alignment_loss" in optimization_logs.keys():
            writer.add_scalar("alignment_loss", optimization_logs["alignment_loss"], global_step=self.global_step)

    def is_step_logged(self, step):
        return step % self.logging_interval == 0

    @classmethod
    def from_config(cls, ds_sizes, config, epoch=0, global_step=0):
        return cls(
            logging_interval=config.logging_interval,
            model_dir=config.model_dir,
            ds_sizes=ds_sizes,
            epoch=epoch,
            global_step=global_step,
        )
