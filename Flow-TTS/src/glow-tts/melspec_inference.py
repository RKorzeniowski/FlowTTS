import re
import shutil
import numpy as np
from pathlib import Path
import torch
import models
import utils
import os
from data_utils import CustomPhonemeMelLoader, TextMelLoader

N_SAMPLES = 6
SYNTH_PARMS = [
    (0.333, 1.0),
    (0.5, 1.0),
    (0.667, 1.0),
    (1.0, 1.0),
    (0.667, 0.75),
    (0.667, 1.25),
]
EXTERNAL_ALIGNMENT = False

DATA_PATH = Path("/home/ec2-user/renard/data/")
MODEL_DIR = "./logs/"
GT_TARGET_PATH = DATA_PATH/"spectrograms_npz_gt"
PROMPTS_PATH = "prompts.txt"
NPZ_FOLDER = "spectrograms_npz_noise_{}_length_{}"
IMG_FOLDER = "spectrograms_img_noise_{}_length_{}"
GT_SOURCE_PATH = Path("/home/ec2-user/renard/data/mf")
GT_SUFFIX = ".npz"
SOS_IDX = 4
SPACE_IDX = 3
PHONEME_KEY = 'name'
SPACE_PADDING = False

SYNTH_FROM_FILE = False
SYNTH_FROM_LIST = False

SOURCE_PATH = Path("/source/path")
LF_FOLDER = "lf_s2s"
FILE_PATH = SOURCE_PATH / PROMPTS_PATH
PHONEM_SAMPLE_PATHS_LIST = [
    SOURCE_PATH/"dummy.npz",
]


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def rescale_melspec(y, transform):
    if transform is not None:
        y /= transform["scale"]
        y -= transform["bias"]
        y = np.clip(y, 0.0, 7.6)
    return y


def prepare_paths(*paths):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_spec_img(y, sample_name, spec_img_path):
    save_img_path = spec_img_path / sample_name
    utils.plot_spectrogram_to_numpy(y, save_path=save_img_path)


def save_spec_npz(y, sample_name, spec_npz_path):
    save_npz_path = spec_npz_path / sample_name
    npz_gen_tst = np.moveaxis(y, 0, -1)
    with open(save_npz_path.with_suffix(".npz"), "wb") as f:
        np.savez(f, spectrogram=npz_gen_tst)


def save_txt(txt, idx, sample_name, data_path, txt_file):
    save_path = data_path / txt_file
    with open(save_path, "a") as f:
        record = f"{idx} {sample_name} {txt}\n"
        f.write(record)


def clean_prompts_txt(data_path, txt_file):
    save_path = data_path / txt_file
    with open(save_path, "w") as f:
        pass


def load_model(model_dir, hps):
    model = models.FlowGenerator(
        utils.get_vocab_size(hps.data.vocab_type, hps.data.vocab_file),
        out_channels=hps.data.n_mel_channels,
        external_alignment=EXTERNAL_ALIGNMENT,
        **hps.model
    ).to("cuda")

    checkpoint_path = utils.latest_checkpoint_path(model_dir)
    utils.load_checkpoint(checkpoint_path, model, n_gpu_bs=hps.train.batch_size)
    model.decoder.store_inverse()
    _ = model.eval()
    return model


def get_single_sample(idx, dataset):
    txt = dataset.get_prompt(idx)
    name = dataset.file_names[idx]
    sequence = dataset[idx][0]

    if SPACE_PADDING:
        sequence = torch.nn.functional.pad(sequence[1:], [1, 1], value=SPACE_IDX)
        sequence = torch.cat([torch.IntTensor([SOS_IDX]), sequence[1:]])

    x_tst = torch.autograd.Variable(sequence).cuda().long().unsqueeze(0)
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()

    if SPACE_PADDING:
        x_tst_lengths += 2

    return x_tst, x_tst_lengths, txt, name


def save_gt(sample_name, source_path, target_path):
    source_path = (source_path / sample_name).with_suffix(GT_SUFFIX)
    target_path = (target_path / sample_name).with_suffix(GT_SUFFIX)
    shutil.copy(str(source_path), str(target_path))


def generate_sample(x_tst, x_tst_lengths, noise_scale, length_scale, sample_name, hps, spec_npz_path, spec_img_path=None, txt=None):
    with torch.no_grad():
        (y_gen_tst, *r), attn_gen, *_ = model(
            x=x_tst,
            x_lengths=x_tst_lengths,
            gen=True,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )
        y_gen_tst = y_gen_tst.squeeze().cpu().numpy()
        y_gen_tst = rescale_melspec(y_gen_tst, transform=hps.data.get("spec_transform"))

        save_spec_npz(y=y_gen_tst, sample_name=sample_name, spec_npz_path=spec_npz_path)
        if txt:
            save_txt(txt=txt, idx=i, sample_name=sample_name, data_path=DATA_PATH, txt_file=PROMPTS_PATH)
        if spec_img_path:
            save_spec_img(y=y_gen_tst, sample_name=sample_name, spec_img_path=spec_img_path)



prepare_paths(GT_TARGET_PATH)
clean_prompts_txt(data_path=DATA_PATH, txt_file=PROMPTS_PATH)

hps = utils.get_hparams_from_dir(MODEL_DIR)
model = load_model(model_dir=MODEL_DIR, hps=hps)
if hps.train.loader_type == "custom":
    val_dataset = CustomPhonemeMelLoader(
        root_dir=hps.data.root_dir,
        sample_names_path=hps.data.eval_files,
        phonemes_folder=hps.data.phonemes_folder,
        mel_folder=hps.data.mel_folder,
        suffix=hps.data.suffix,
        spec_transform=hps.data.get("spec_transform"),
    )
elif hps.train.loader_type == "lj":
    val_dataset = TextMelLoader(hps.data.validation_files, hps.data)

if SYNTH_FROM_LIST:
    for path in PHONEM_SAMPLE_PATHS_LIST:
        phonemes = torch.LongTensor(np.load(path)[PHONEME_KEY]).unsqueeze(0).cuda()
        x_len = torch.LongTensor([phonemes.shape[1]]).cuda()
        spec_npz_path = DATA_PATH / "spectrograms_npz_list_noise_0.667_length_1.0"
        prepare_paths(spec_npz_path)
        sample_name = path.stem
        generate_sample(x_tst=phonemes, x_tst_lengths=x_len, noise_scale=0.667,
                        length_scale=1., sample_name=sample_name, spec_npz_path=spec_npz_path, hps=hps)




else:
    for noise_scale, length_scale in SYNTH_PARMS:
        spec_img_path = DATA_PATH / IMG_FOLDER.format(noise_scale, length_scale)
        spec_npz_path = DATA_PATH / NPZ_FOLDER.format(noise_scale, length_scale)
        prepare_paths(spec_img_path, spec_npz_path)

        if SYNTH_FROM_FILE:
            pattern = r'\<.*?\>'
            file_names, prompts = [], []
            with open(FILE_PATH, "r") as f:
                for line in f:
                    file_name, prompt = line.strip("( ").split(maxsplit=1)
                    clean_prompt = re.sub(pattern, '', prompt).strip(" )\n").strip('"')
                    path = (SOURCE_PATH / LF_FOLDER / file_name).with_suffix(".npz")
                    phonemes = torch.LongTensor(np.load(path)[PHONEME_KEY]).unsqueeze(0).cuda()
                    x_len = torch.LongTensor([phonemes.shape[1]]).cuda()
                    sample_name = path.stem
                    generate_sample(x_tst=phonemes, x_tst_lengths=x_len, noise_scale=noise_scale,
                                    length_scale=length_scale, sample_name=sample_name, spec_npz_path=spec_npz_path, hps=hps)



        else:
            for i in range(N_SAMPLES):
                x_tst, x_tst_lengths, txt, sample_name = get_single_sample(idx=i, dataset=val_dataset)
                generate_sample(x_tst, x_tst_lengths, noise_scale, length_scale,
                                sample_name, hps, spec_npz_path, spec_img_path, txt)
                save_gt(sample_name=sample_name, source_path=GT_SOURCE_PATH, target_path=GT_TARGET_PATH)