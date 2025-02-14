import abc
import random

import json
import numpy as np
from pathlib import Path
import re
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from scipy.io.wavfile import read

from data_processing import samplers
from utils import training_utils
from data_processing.text import cmudict
from data_processing.text import symbols as ljsymbols
from data_processing.text.text import text_to_sequence
from copy import deepcopy


def single_letter_phoneme_filter(config, sub_dataset):
    phoneme_indexes = deepcopy(config.phoneme_indexes)
    filtered_sample_paths = []
    for idx in range(len(sub_dataset.sample_paths)):
        phonemes, melspec, alignment = sub_dataset[idx]
        for p_idx in phoneme_indexes.keys():
            if phoneme_indexes[p_idx] > 0 and p_idx in phonemes:
                filtered_sample_paths.append(sub_dataset.sample_paths[idx])
                phoneme_indexes[p_idx] -= 1
            if sum(phoneme_indexes.values()) == 0:
                return filtered_sample_paths

    return filtered_sample_paths


def format_alignment(alignment):
    alignment_list = [int(v) for v in alignment.split()]
    return alignment_list


class Databunch:
    def __init__(self, datasets, vocab):
        self.vocab = vocab
        self.datasets = datasets

    def get_datasets_sizes(self):
        return {name: len(dl) for name, dl in self.datasets.items()}

    def __getitem__(self, key):
        return self.datasets[key]

    def get_vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def get_vocab(vocab_file, phoneme_key):
        with open(vocab_file) as json_file:
            symbols = json.load(json_file)
            if phoneme_key in symbols:
                symbols = symbols[phoneme_key]
        return symbols

    @classmethod
    def from_config(cls, config, hparams, bs, rank, inference=False):
        if config.ds_type == "lj":
            return cls.from_config_lj(config, hparams, bs, rank)
        elif "custom" in config.ds_type:
            return cls.from_config_custom(config, hparams, inference, bs, rank)
        elif config.ds_type == "test":
            return cls.from_config_test(config)
        elif config.ds_type == "single_letter":
            return cls.from_config_single_letter(config, hparams)
        else:
            raise ValueError()

    @classmethod
    def from_config_test(cls, config):
        samples = []
        if config.type == "pattern":
            idxes = torch.randint(0, len(config.mean), config.sample_shapes)
            for _ in range(config.sample_count):
                sample = torch.zeros(config.sample_shapes)
                for i in range(len(config.mean)):
                    sample += (idxes == i).type(torch.FloatTensor) * (torch.randn(config.sample_shapes) * config.std[i] + config.mean[i])
                samples.append((sample, idxes))

        elif config.type == "gaussians":
            for _ in range(config.sample_count):
                idx = random.randint(0, len(config.mean) - 1)
                idxes = torch.zeros(config.sample_shapes) + idx
                sample = torch.randn(config.sample_shapes) * config.std[idx] + config.mean[idx]
                samples.append((sample, idxes))

        elif config.type == "time_dependant":
            frames = config.sample_shapes[2] // config.sub_dist_count
            for _ in range(config.sample_count):
                sample = torch.randn(config.sample_shapes)
                idxes = torch.zeros(config.sample_shapes)
                for i in range(1, config.sub_dist_count):
                    idx = random.randint(0, len(config.mean) - 1)
                    start = frames * (i - 1)
                    end = frames * i
                    sample[:, :, start:end] *= config.std[idx]
                    sample[:, :, start:end] += config.mean[idx]
                    idxes[:, :, start:end] = idx
                samples.append((sample, idxes))

        elif config.type == "channel_dependant":
            channels = config.sample_shapes[1] // config.sub_dist_count
            for _ in range(config.sample_count):
                sample = torch.randn(config.sample_shapes)
                idxes = torch.zeros(config.sample_shapes)
                for i in range(1, config.sub_dist_count):
                    idx = random.randint(0, len(config.mean) - 1)
                    start = channels * (i - 1)
                    end = channels * i
                    sample[:, start:end, :] *= config.std[idx]
                    sample[:, start:end, :] += config.mean[idx]
                    idxes[:, start:end, :] = idx
                samples.append((sample, idxes))

        elif config.type == "mixture_gaussians":
            for _ in range(config.sample_count):
                sample = torch.zeros(config.sample_shapes)
                idxes = torch.randint(0, len(config.mean), config.sample_shapes)
                for i in range(len(config.mean)):
                    sample += (idxes == i).type(torch.FloatTensor) * (
                                torch.randn(config.sample_shapes) * config.std[i] + config.mean[i])
                samples.append((sample, idxes))
        else:
            raise ValueError()

        return cls(datasets={"train": samples}, vocab=None)

    @classmethod
    def from_config_lj(cls, config, hparams, bs, rank):
        if hparams.bucket_samples:
            if hparams.distributed:
                train_sampler = samplers.DistributedBySequenceLengthSampler
                eval_sampler = samplers.DistributedBySequenceLengthSampler
            else:
                train_sampler = samplers.RandomBySequenceLengthSampler
                eval_sampler = samplers.RandomBySequenceLengthSampler
        else:
            train_sampler = torch.utils.data.RandomSampler
            eval_sampler = torch.utils.data.RandomSampler

        datasets_params = [
            ['train', config.train_files, train_sampler, {"batch_size": bs}],
        ]
        if rank == 0:
            datasets_params.append(['eval', config.eval_files, eval_sampler, {"batch_size": bs}])

        if hparams.distributed:
            for x in datasets_params:
                x[3]["rank"] = rank

        datasets = {}
        for name, audiopaths_and_text, sampler_fn, smpl_kwargs in datasets_params:
            train_dataset = TextMelLoader(
                hparams=hparams,
                audiopaths_and_text=audiopaths_and_text,
                text_cleaners=config.text_cleaners,
                cmudict_path=config.cmudict_path,
                maxpool=hparams.maxpool,
            )

            sampler = sampler_fn(
                data_source=train_dataset,
                max_spec_cutoff_len=hparams.max_spec_cutoff_len,
                min_spec_cutoff_len=hparams.min_spec_cutoff_len,
                batch_buckets=hparams.batch_buckets,
                **smpl_kwargs
            )
            collate_fn = TextMelCollate(alignment=True, n_frames_per_step=hparams.n_sqz)
            loader = DataLoader(
                train_dataset,
                num_workers=hparams.loader_workers,
                shuffle=False,
                batch_size=bs,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            datasets[name] = loader

        return cls(datasets=datasets, vocab=ljsymbols.symbols)

    @classmethod
    def from_config_custom(cls, config, hparams, inference, bs, rank):
        vocab = cls.get_vocab(config.vocab_file, config.phoneme_key)
        if hparams.bucket_samples:
            if hparams.distributed:
                train_sampler = samplers.DistributedBySequenceLengthSampler
                eval_sampler = samplers.DistributedBySequenceLengthSampler
            else:
                train_sampler = samplers.RandomBySequenceLengthSampler
                eval_sampler = samplers.SequentialSampler
        else:
            train_sampler = samplers.RandomSampler
            eval_sampler = samplers.RandomSampler

        datasets_params = [
            ['train', config.train_files, train_sampler, {"batch_size": bs}],
        ]
        if rank == 0:
            datasets_params.append(['eval', config.eval_files, eval_sampler, {"batch_size": bs}])
            if inference:
                datasets_params.append(['inference', config.inference_files, eval_sampler, {"batch_size": bs}])

        if hparams.distributed:
            for x in datasets_params:
                x[3]["rank"] = rank

        if config.multiclass:
            dataset = CustomAlignPhonemeMulticlassMelLoader
        else:
            dataset = CustomAlignedPhonemeMelLoader

        datasets = {}
        for name, data_path, sampler_fn, smpl_kwargs in datasets_params:
            sub_dataset = dataset(
                data_paths=data_path,
                mel_folder=config.mel_folder,
                suffix=config.suffix,
                spec_transform=config.spec_transform,
                phoneme_key=config.phoneme_key,
                trim_silence=hparams.trim_silence,
                add_noise=hparams.add_noise,
                noise_max_value=hparams.noise_max_value,
            )
            if name != "train":
                sampler = sampler_fn(
                    data_source=sub_dataset,
                    max_spec_cutoff_len=hparams.max_spec_cutoff_len,
                    min_spec_cutoff_len=hparams.min_spec_cutoff_len,
                    batch_buckets=hparams.batch_buckets,
                )
            else:
                sampler = sampler_fn(
                    data_source=sub_dataset,
                    max_spec_cutoff_len=hparams.max_spec_cutoff_len,
                    min_spec_cutoff_len=hparams.min_spec_cutoff_len,
                    batch_buckets=hparams.batch_buckets,
                    **smpl_kwargs
                )

            align = config.alignment if name != 'inference' else True
            collate_fn = TextMelCollate(alignment=align, n_frames_per_step=hparams.n_sqz)
            loader = DataLoader(
                sub_dataset,
                num_workers=hparams.loader_workers,
                shuffle=False,
                batch_size=bs,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            datasets[name] = loader

        return cls(datasets=datasets, vocab=vocab)

    @classmethod
    def from_config_single_letter(cls, config, hparams):
        vocab = cls.get_vocab(config.vocab_file)
        train_sampler = torch.utils.data.RandomSampler
        eval_sampler = torch.utils.data.SequentialSampler
        datasets_params = [
            ('train', config.train_files, train_sampler),
            ('eval', config.eval_files, eval_sampler)
        ]

        datasets = {}
        for name, data_path, sampler_fn in datasets_params:
            sub_dataset = CustomAlignedPhonemeMelLoader(
                data_paths=data_path,
                mel_folder=config.mel_folder,
                suffix=config.suffix,
                spec_transform=config.spec_transform,
            )

            sub_dataset.sample_paths = single_letter_phoneme_filter(config, sub_dataset)

            sampler = sampler_fn(
                data_source=sub_dataset,
                max_spec_cutoff_len=hparams.max_spec_cutoff_len,
                min_spec_cutoff_len=hparams.min_spec_cutoff_len,
                batch_buckets=hparams.batch_buckets,
            )
            collate_fn = TextMelCollate(alignment=True, n_frames_per_step=hparams.n_sqz)
            loader = DataLoader(
                sub_dataset,
                num_workers=hparams.loader_workers,
                shuffle=False,
                batch_size=hparams.batch_size,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            datasets[name] = loader

        return cls(datasets=datasets, vocab=vocab)


class BaseCustomPhonemeMelLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths,
        mel_folder,
        suffix,
        spec_transform,
        phoneme_key,
        trim_silence,
        add_noise,
        noise_max_value,
        split_pattern=r'\<.*?\>',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.split_pattern = split_pattern
        self.spec_transform = spec_transform
        self.phoneme_key = phoneme_key
        self.trim_silence = trim_silence
        self.add_noise = add_noise
        self.noise_max_value = noise_max_value

        self.sample_paths = self.get_records(
            data_paths=data_paths,
            mel_folder=mel_folder,
            suffix=suffix,
        )

    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    @abc.abstractmethod
    def get_record(self, line, root_path, mel_folder, phonemes_folder, suffix, **kwargs):
        pass

    def get_melspec(self, path, name='spectrogram'):
        melspec = torch.from_numpy(np.load(path)[name]).permute((1, 0))
        if self.trim_silence:
            melspec = training_utils.trim_silence(melspec)
        if self.add_noise:
            melspec = melspec + torch.rand_like(melspec) * self.noise_max_value
        if self.spec_transform is not None:
            melspec = melspec + self.spec_transform["bias"]
            melspec = melspec * self.spec_transform["scale"]
        return melspec

    def get_phonemes(self, path):
        file = np.load(path)
        if "phonemes" in list(file.keys()):
            key = "phonemes"
        else:
            key = self.phoneme_key
        numericized_phonemes = torch.LongTensor(file[key])
        return numericized_phonemes

    def __len__(self):
        return len(self.sample_paths)

    def get_records(self, data_paths, mel_folder, suffix):
        records = []
        data_paths = deepcopy(data_paths)
        for data_src in data_paths:
            root_path = Path(data_src.pop("root_dir"))
            prompts_file = data_src.pop("prompts_file")
            phonemes_folder = data_src.pop("phonemes_folder")
            with open(root_path / prompts_file, "r") as f:
                for line in f:
                    if data_src["class"] in line:
                        record = self.get_record(
                            line=line,
                            root_path=root_path,
                            mel_folder=mel_folder,
                            phonemes_folder=phonemes_folder,
                            suffix=suffix,
                            **data_src
                        )
                        records.append(record)
        return records

    def get_filename_and_prompt(self, line):
        file_name, prompt = line.strip("( ").split(maxsplit=1)
        clean_prompt = re.sub(self.split_pattern, '', prompt).strip(" )\n").strip('"')
        return file_name, clean_prompt

    @staticmethod
    def get_path(root_path, folder, file_name, suffix):
        root_path = Path(root_path)
        path = root_path / folder / file_name
        path = path.with_suffix(suffix)
        return str(path)


class CustomPhonemeMelLoader(BaseCustomPhonemeMelLoader):
    def __getitem__(self, index):
        melspec_path, phonemes_path, prompt = self.sample_paths[index]
        phonemes = self.get_phonemes(phonemes_path)
        melspec = self.get_melspec(melspec_path)
        return phonemes, melspec

    def get_record(self, line, root_path, mel_folder, phonemes_folder, suffix, **kwargs):
        file_name, prompt = self.get_filename_and_prompt(line)
        spec_path = self.get_path(root_path, mel_folder, file_name, suffix)
        phonem_path = self.get_path(root_path, phonemes_folder, file_name, suffix)
        return spec_path, phonem_path, prompt


class CustomAlignedPhonemeMelLoader(BaseCustomPhonemeMelLoader):
    def __init__(self, **kwargs):
        self.alignment_mapping = {}
        self.alignment_root = ""
        super().__init__(**kwargs)

    def __getitem__(self, index):
        melspec_path, phonemes_path, alignment, prompt = self.sample_paths[index]
        phonemes = self.get_phonemes(phonemes_path)
        melspec = self.get_melspec(melspec_path)
        return phonemes, melspec, alignment

    def get_record(self, line, root_path, mel_folder, phonemes_folder, suffix, **kwargs):
        file_name, prompt = self.get_filename_and_prompt(line)
        spec_path = self.get_path(root_path, mel_folder, file_name, suffix)
        phonem_path = self.get_path(root_path, phonemes_folder, file_name, suffix)
        if self.alignment_root != root_path:
            self.populate_alignment_mapping(root_path, kwargs["alignment_path"])
        alignment = self.alignment_mapping[file_name]
        return spec_path, phonem_path, alignment, prompt

    def populate_alignment_mapping(self, root_path, alignment_path):
        with open(root_path / alignment_path, 'r') as f:
            for line in f:
                file_name, alignment = line.strip("( ").split(maxsplit=1)
                alignment_list = format_alignment(alignment)
                self.alignment_mapping[file_name] = alignment_list
            self.alignment_root = root_path


class CustomPhonemeMulticlassMelLoader(BaseCustomPhonemeMelLoader):
    def __getitem__(self, index):
        melspec_path, phonemes_path, prompt, class_idx, class_peak_value = self.sample_paths[index]
        phonemes = self.get_phonemes(phonemes_path)
        melspec = self.get_melspec(melspec_path)
        return phonemes, melspec, class_idx, class_peak_value

    def get_record(self, line, root_path, mel_folder, phonemes_folder, suffix, **kwargs):
        file_name, prompt = self.get_filename_and_prompt(line)
        spec_path = self.get_path(root_path, mel_folder, file_name, suffix)
        phonem_path = self.get_path(root_path, phonemes_folder, file_name, suffix)
        class_idx = kwargs['class_idx']
        class_peak_value = kwargs['class_peak_value']
        return spec_path, phonem_path, prompt, class_idx, class_peak_value


class CustomAlignPhonemeMulticlassMelLoader(BaseCustomPhonemeMelLoader):
    def __init__(self, **kwargs):
        self.alignment_mapping = {}
        self.alignment_root = ""
        super().__init__(**kwargs)

    def __getitem__(self, index):
        melspec_path, phonemes_path, alignment, prompt, class_idx, class_peak_value = self.sample_paths[index]
        phonemes = self.get_phonemes(phonemes_path)
        melspec = self.get_melspec(melspec_path)
        return phonemes, melspec, alignment, class_idx, class_peak_value

    def get_record(self, line, root_path, mel_folder, phonemes_folder, suffix, **kwargs):
        file_name, prompt = self.get_filename_and_prompt(line)
        spec_path = self.get_path(root_path, mel_folder, file_name, suffix)
        phonem_path = self.get_path(root_path, phonemes_folder, file_name, suffix)
        class_idx = kwargs['class_idx']
        class_peak_value = kwargs['class_peak_value']
        if self.alignment_root != root_path:
            self.populate_alignment_mapping(root_path, kwargs["alignment_path"])
        alignment = self.alignment_mapping[file_name]
        return spec_path, phonem_path, alignment, prompt, class_idx, class_peak_value

    def get_class_subset(self, class_idx):
        subset_samples = []
        for i in range(len(self.sample_paths)):
            _, _, _, _, sample_class_idx, _ = self.sample_paths[i]
            if sample_class_idx == class_idx:
                subset_samples.append(self.sample_paths[i])
        return subset_samples

    def populate_alignment_mapping(self, root_path, alignment_path):
        with open(root_path / alignment_path, 'r') as f:
            for line in f:
                file_name, alignment = line.strip("( ").split(maxsplit=1)
                alignment_list = format_alignment(alignment)
                self.alignment_mapping[file_name] = alignment_list
            self.alignment_root = root_path


class CustomAlignedF0EnergyPhonemeMelLoader(BaseCustomPhonemeMelLoader):
    def __init__(self, **kwargs):
        self.alignment_mapping = {}
        self.alignment_root = ""
        super().__init__(**kwargs)

    def get_f0(self):
        pass

    def get_energy(self):
        pass

    def __getitem__(self, index):
        melspec_path, phonemes_path, f0_path, energy_path, alignment, prompt = self.sample_paths[index]
        phonemes = self.get_phonemes(phonemes_path)
        melspec = self.get_melspec(melspec_path)
        f0 = self.get_f0(f0_path)
        energy = self.get_energy(energy_path)
        return phonemes, melspec, f0, energy, alignment

    def get_record(self, line, root_path, mel_folder, phonemes_folder, suffix, **kwargs):
        file_name, prompt = self.get_filename_and_prompt(line)
        spec_path = self.get_path(root_path, mel_folder, file_name, suffix)
        phonem_path = self.get_path(root_path, phonemes_folder, file_name, suffix)
        f0_path = self.get_path(root_path, kwargs["f0_folder"], file_name, suffix)
        energy_path = self.get_path(root_path, kwargs["energt_folder"], file_name, suffix)
        if self.alignment_root != root_path:
            self.populate_alignment_mapping(root_path, kwargs["alignment_path"])
        alignment = self.alignment_mapping[file_name]
        return spec_path, phonem_path, f0_path, energy_path, alignment, prompt

    def populate_alignment_mapping(self, root_path, alignment_path):
        with open(root_path / alignment_path, 'r') as f:
            for line in f:
                file_name, alignment = line.strip("( ").split(maxsplit=1)
                alignment_list = format_alignment(alignment)
                self.alignment_mapping[file_name] = alignment_list
            self.alignment_root = root_path


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams, audiopaths_and_text, text_cleaners, cmudict_path, maxpool):
        self.text_cleaners = text_cleaners
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.audiopaths_and_text = self.load_filepaths_and_text(audiopaths_and_text)
        self.stft = training_utils.TacotronSTFT(hparams)
        self.maxpool = torch.nn.MaxPool1d(maxpool)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def load_filepaths_and_text(self, filename, mel_folder="mf", split="|"):
        with open(filename, encoding='utf-8') as f:
            filepaths_and_text = []
            for line in f:
                path, text = line.strip().split(split)
                path = path.replace(".wav", ".npz")
                path = path.replace("wavs", mel_folder)
                text_norm = torch.IntTensor(
                    text_to_sequence(text, self.text_cleaners, self.cmudict)
                )
                filepaths_and_text.append((path, text_norm))
        return filepaths_and_text

    def get_mel_text_pair(self, audiopath_and_text):
        melpath, text = audiopath_and_text[0], audiopath_and_text[1]
        mel = self.get_mel(melpath)
        return text, mel

    def load_wav_to_torch(self, full_path):
        sampling_rate, data = read(full_path)
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate

    def get_mel(self, filename):
        melspec = torch.from_numpy(np.load(filename)["spectrogram"]).permute(1, 0)
        with torch.no_grad():
            melspec = self.maxpool(melspec.permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        return melspec

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, alignment, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step
        self.alignment = alignment

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        outputs = [text_padded, input_lengths, mel_padded, output_lengths]
        if len(batch[0]) >= 3:
            align = list(zip(*batch))[2]
            align = [torch.LongTensor(a) for a in align]
            align_padded = torch.nn.utils.rnn.pad_sequence(align, batch_first=True)
            align_padded = align_padded[ids_sorted_decreasing]
            align_lengths = torch.LongTensor([x.shape[0] for x in align])[ids_sorted_decreasing]
            comb_alignment = unpack_alignment(align_padded, align_lengths, div=self.n_frames_per_step)
            outputs.extend(comb_alignment)

        if len(batch[0]) >= 4:
            cls_idx, peak_idx = 3, 4

            classes = list(zip(*batch))[cls_idx]
            classes = torch.LongTensor(classes)
            outputs.append(classes)

            peaks = list(zip(*batch))[peak_idx]
            peaks = torch.FloatTensor(peaks)
            outputs.append(peaks)

        return outputs


def unpack_alignment(external_alignment, align_phoneme_lengths, div):
  bs = external_alignment.shape[0]
  align_mel_lengths = external_alignment.sum(dim=1)
  max_mel_length, max_mel_idx = align_mel_lengths.max(0)
  if max_mel_length % div:
    align_mel_lengths[max_mel_idx] += div - max_mel_length % div
    max_mel_length = align_mel_lengths[max_mel_idx]
  formatted_alignment = torch.zeros((bs, 1, max(align_phoneme_lengths), max_mel_length))

  "needs more efficient implementation"
  for bs_idx in range(bs):
    sample_alignment = external_alignment[bs_idx]
    cum_sum = torch.cumsum(sample_alignment, dim=0)
    phonem_align_range = sum(sample_alignment)

    phonem_idx = 0
    for i in range(phonem_align_range):
      while cum_sum[phonem_idx] <= i:
        phonem_idx += 1
      formatted_alignment[bs_idx, 0, phonem_idx, i] = 1

  return formatted_alignment, align_mel_lengths
