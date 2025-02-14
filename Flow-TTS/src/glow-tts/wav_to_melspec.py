import utils
import commons
import torch
from utils import load_wav_to_torch
from pathlib import Path
import numpy as np

hparams = utils.get_hparams().data

stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

root = Path("/home/ec2-user/renard/data/no-attention-alignment-cleaned")
wav_root = root/"wav"
txt_path = root/"texts.txt"
target_mel_path = root/"mf_glow"


with open(txt_path, "r") as f:
    for line in f:
        file_name = line.strip("( ").split(maxsplit=1)[0]
        audio_path = (wav_root / file_name).with_suffix(".wav")
        audio, sampling_rate = load_wav_to_torch(audio_path)

        if sampling_rate != stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, stft.sampling_rate))

        audio = audio + torch.rand_like(audio) / 2
        audio_norm = audio / hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0).permute((1, 0)).numpy()

        target_mel_file_path = (target_mel_path / file_name).with_suffix(".npz")
        with open(str(target_mel_file_path), "wb") as f:
            np.savez(f, spectrogram=melspec)
