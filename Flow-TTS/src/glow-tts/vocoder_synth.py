import os
import subprocess
from os import listdir
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

SPECS_PATH = Path("/home/ec2-user/renard/data/preds/VAE_neutral_down_latVAE_newloss_context_smoothing")
FOLDER_NAME_FILTER = "pred_specs"
TARGET_FOLDER_PREFIX = "eval"
SYNTH_COMMAND = "bash synth_student.sh {source_path} {target_path}"
SYNTH_WITH_WAVEGLOW = False
WAVEGLOW_PATH = './waveglow/waveglow_256channels_ljs_v3.pt'
SR = 22050

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def load_waveglow(checkpoint_path):
    sys.path.append('./waveglow/')
    waveglow = torch.load(checkpoint_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    _ = waveglow.cuda().eval()
    waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")
    return waveglow


def synth_sample(spec, sigma=.666):
    try:
        audio = waveglow.infer(spec.half(), sigma=sigma)
    except:
        audio = waveglow.infer(spec, sigma=sigma)
    return audio[0].clamp(-1, 1).data.cpu().float().numpy()


if SYNTH_WITH_WAVEGLOW:
    import torch
    from apex import amp
    import librosa
    waveglow = load_waveglow(WAVEGLOW_PATH)

    for folder_name in listdir(SPECS_PATH):
        if FOLDER_NAME_FILTER in folder_name:
            spec_folder = SPECS_PATH / folder_name
            for file_name in tqdm(listdir(spec_folder)):
                spec_path = SPECS_PATH/folder_name / file_name
                spec = torch.from_numpy(np.load(spec_path)['spectrogram']).permute((1, 0))
                spec = spec.unsqueeze(0).cuda()
                wav = synth_sample(spec)
                target_path = SPECS_PATH / (TARGET_FOLDER_PREFIX + folder_name.replace(FOLDER_NAME_FILTER, ""))
                target_path.mkdir(exist_ok=True, parents=True)
                target_path = (target_path / file_name).with_suffix(".wav")
                librosa.output.write_wav(target_path, wav, sr=SR)

else:
    os.chdir("/home/ec2-user/uv_pw_inference")
    for folder_name in listdir(SPECS_PATH):
        if FOLDER_NAME_FILTER in folder_name:
            source_path = SPECS_PATH / folder_name
            target_path = SPECS_PATH / (TARGET_FOLDER_PREFIX + folder_name.replace(FOLDER_NAME_FILTER, ""))
            command = SYNTH_COMMAND.format(source_path=source_path, target_path=target_path)

            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
