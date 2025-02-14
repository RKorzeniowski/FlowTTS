import random

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn
from utils.audio_processing import dynamic_range_compression
from utils.audio_processing import dynamic_range_decompression
from utils.stft import STFT
from scipy.io.wavfile import read

import hparams as hp

MATPLOTLIB_FLAG = False


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        self.__dict__[key] = value


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class TacotronSTFT(torch.nn.Module):
  def __init__(self, hparams):
    super(TacotronSTFT, self).__init__()
    self.n_mel_channels = hparams.n_mel_channels
    self.sampling_rate = hparams.sampling_rate
    self.stft_fn = STFT(hparams.fft_size, hparams.hop_size, hparams.win_length)
    mel_basis = librosa_mel_fn(
        hparams.sampling_rate, hparams.fft_size, hparams.n_mel_channels, hparams.mel_fmin, hparams.mel_fmax
    )
    mel_basis = torch.from_numpy(mel_basis).float()
    self.register_buffer('mel_basis', mel_basis)

  def spectral_normalize(self, magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

  def spectral_de_normalize(self, magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output

  def mel_spectrogram(self, y):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)

    magnitudes, phases = self.stft_fn.transform(y)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(self.mel_basis, magnitudes)
    mel_output = self.spectral_normalize(mel_output)
    return mel_output


def trim_silence(mel_spec, cutoff=0.05):
    mel_dim, time_dim = mel_spec.shape
    value = torch.sum(mel_spec, dim=0)
    start_silence_mask = torch.cumsum(value > cutoff, dim=0) != 0
    end_silence = torch.cumsum(value > cutoff, dim=0)
    max_non_silence_idx = end_silence.max()
    end_silence_mask = end_silence != max_non_silence_idx
    silence_mask = (start_silence_mask & end_silence_mask).repeat(mel_dim, 1)
    mel_spec = mel_spec[silence_mask].reshape((mel_dim, -1))
    return mel_spec


def clip_grad_value(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

        p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def split(x, split_point):
    return x[:, :split_point, :], x[:, split_point:, :]


def get_mask_from_lens(lens):
    max_length = lens.max()
    x = torch.arange(max_length, dtype=lens.dtype, device=lens.device)
    mask = x.unsqueeze(0) < lens.unsqueeze(1)
    return torch.unsqueeze(mask, 1)


def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_spectrogram_to_numpy(spectrogram, save_path=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots()
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    if save_path is None:
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
    else:
        plt.savefig(save_path)
        data = None

    return data
