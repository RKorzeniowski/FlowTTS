import torch
from data_processing import datasets
from pathlib import Path
from pipeline import pipeline
import pickle

checkpoint_path = "/home/ec2-user/renard/Flow-TTS/src/flow_tts/logs/checkpoint_model.pth"
experiments_folder = "/home/ec2-user/renard/preds/experiments"
experiments_folder = Path(experiments_folder)

tp = pipeline.VAETrainPipeline.from_checkpoint(checkpoint_path, 0, 1)
# target_cls = 0
mode = 'train'

def load_data(i, target_cls):
    samples = tp.databunch[mode].dataset.get_class_subset(target_cls)

    melspec_path, phonemes_path, alignment, prompt, class_idx, class_peak_value = samples[i]

    text = tp.databunch[mode].dataset.get_phonemes(path=phonemes_path).unsqueeze(0).to(0)
    text_len = torch.LongTensor([text.shape[1]]).to(0)

    spec = tp.databunch[mode].dataset.get_melspec(path=melspec_path)
    spec_len = torch.LongTensor([spec.shape[1]])

    alignment = [torch.LongTensor(alignment)]
    alignment = torch.nn.utils.rnn.pad_sequence(alignment, batch_first=True)
    align_lengths = torch.LongTensor([x.shape[0] for x in alignment])
    alignment, align_lengths = datasets.unpack_alignment(alignment, align_lengths, div=tp.hparams.n_sqz)
    alignment = alignment.to(0)
    return spec, spec_len, text, text_len, alignment, class_idx, class_peak_value


def calculate_clusters():
    # calculate mean and std of all specific class spectrograms VAE
    class_centroids = {}
    for name_cls, target_cls in [("neutral", 0), ("news", 1), ("conversational", 2)]:
        sample_count = len(tp.databunch[mode].dataset.get_class_subset(target_cls))
        sum_mu, sum_s = 0, 0
        for i in range(sample_count):
            spec, spec_len, text, text_len, alignment, cls_idx, class_peak_value = load_data(i=i, target_cls=target_cls)
            spec = spec.unsqueeze(0)
            z, mu, logs = tp.model.VAE(spec.cuda())
            sum_mu += mu
            sum_s += torch.exp(logs)
        class_centroids[target_cls] = {"mu": sum_mu.cpu().data.numpy() / sample_count, "std": sum_s.cpu().data.numpy() / sample_count}
    print(class_centroids)
    with open('centroids_VAE_all_dcVAE_new12block.pkl', 'wb') as f:
        pickle.dump(class_centroids, f)


calculate_clusters()
