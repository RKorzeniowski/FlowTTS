import torch
from utils import training_utils
from pipeline import pipeline
from data_processing import datasets
import numpy as np
from pathlib import Path
import subprocess
from os import listdir
import os

checkpoint_path = "/home/ec2-user/renard/Flow-TTS/src/flow_tts/logs/checkpoint_model.pth"
experiments_folder = "/home/ec2-user/renard/preds/experiments"
experiments_folder = Path(experiments_folder)

tp = pipeline.VAETrainPipeline.from_checkpoint(checkpoint_path, 0, 1)
# class: 0=neutral, 1=news, 2=conversational
target_cls = 0
mode = 'train'

eps = 1.0
synth_command = "bash vocoder.sh {source_path} {target_path} && source deactivate"
folder_name_filter = "spec_"
target_folder_prefix = "audio_"
vocoder_path = "/home/ec2-user/uv_pw_inference"


def run_audio_pred(data_path):
    cwd = os.getcwd()
    os.chdir(vocoder_path)
    if folder_name_filter in str(data_path):

        target_path = Path(str(data_path).replace(folder_name_filter, target_folder_prefix))
        target_path.mkdir(exist_ok=True)
        command = synth_command.format(source_path=data_path, target_path=target_path)
        subprocess.run(command, shell=True)

    os.chdir(cwd)

def save_spec(spec, path):
    spec = np.moveaxis(spec, 0, -1)
    with open(path.with_suffix(".npz"), "wb") as f:
        np.savez(f, spectrogram=spec)

def get_combined_context_from_spec(spec, spec_len, text, text_len, alignment):
    spec, spec_len = tp.model.decoder.squeezer.trim_dims(spec, lens=spec_len)

    text_mask = training_utils.get_mask_from_lens(text_len)
    text_encodings = tp.model.text_encoder(text, text_mask)
    context_attn, _, spec_len = tp.model.get_context(alignment=alignment,text=text,text_encodings=text_encodings,text_mask=text_mask,spec_len=spec_len)

    spec = spec.cuda()
    vae_context, mu, logs = tp.model.VAE(spec)
    context = tp.model.context_combiner(attn=context_attn, aux_context=vae_context)
    context, spec_len = tp.model.decoder.squeezer.trim_dims(context, lens=spec_len)
    return context, spec_len

def get_Z_from_spec(spec, spec_len, context):
    z, log_det = tp.model.decoder.forward(
        x=spec,
        spec_len=spec_len,
        context_attn=context,
    )
    return z

def predict_spec(z, spec_len, context):

    spec = tp.model.decoder.reverse(z, spec_len=spec_len, context_attn=context)
    return spec

def inference_phoneme_context(text, text_len, alignment):
    text_mask = training_utils.get_mask_from_lens(text_len)
    text_encodings = tp.model.text_encoder(text, text_mask)
    context = tp.model.upsample_alignment(alignment.squeeze(1).transpose(1, 2), text_encodings)
    return context

def inference_combine_prior_context(context, spec_len, cls_peak, std):
    mu = cls_peak
    dist_std = torch.ones_like(cls_peak) * std
    sampled_prior = tp.model.sample_latent_vae_vector(spec_len=spec_len, mu=mu, std=dist_std)
    vae_context = sampled_prior[:, :, 0]
    context = tp.model.context_combiner(attn=context, aux_context=vae_context.cuda())
    return context

def inference_z(text, spec_len, std=1):
    z = tp.model.sample_latent_vector(text, spec_len, std)
    return z

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
####

# context=0 i rozne Z |
    # VAE output
    # text enc
    # Z

# analysis of spectrograms in terms of F0 etc.
# experiment 1: const
# 10 times same samples with low/high std and the same context
def predict_const_Z():
    # from 10 samples
    # content the same, acoustic features change when context fixed and Z changes
    experiment_folder = experiments_folder / f"{folder_name_filter}const_Z_eps_{eps}"
    experiment_folder.mkdir(exist_ok=True, parents=True)
    for j in range(3):
        for i in range(10):
            spec, spec_len, text, text_len, alignment, _, _ = load_data(i=i, target_cls=0)
            spec, spec_len = spec.unsqueeze(0), spec_len.unsqueeze(0)
            # get context
            context, spec_len = get_combined_context_from_spec(spec, spec_len, text, text_len, alignment)
            # Z=0 / Z=0 + eps / fixed_Z (clip to same length?) / fixed_context & sample Z
            spec, spec_len = tp.model.decoder.squeezer.trim_dims(spec, lens=spec_len)
            z = torch.randn_like(spec) * eps
            z = z.cuda()

            # generate spectrogram
            pred_spec = predict_spec(z=z, spec_len=spec_len, context=context)
            # save audio
            sample_name = experiment_folder / f"sample_{i}_run_{j}.npz"
            np_spec = pred_spec[0].cpu().data.numpy()
            save_spec(spec=np_spec, path=sample_name)
    return experiment_folder

# experiment_folder = predict_const_Z()
# run_audio_pred(data_path=experiment_folder)

def predict_const_context():
    # get 10 samples
    # sample VAE noise that matches the longest one and repeat so that it matches for all 10 samples
    # combine with phoneme context
    # predict and compare
    pass

def change_Z_channels():
    # for ch in Z:
    #    spec[ch] *= 10 # different types of noisy speech depending on channel
    #    spec[ch] += 0.3 # no effect
    #    spec[ch, :t//4] *= 10 #
    #    spec[ch] = 0.3 # no effect
    #    spec[ch] = 0. # just bias so sounds good
    experiment_folder = experiments_folder / f"{folder_name_filter}channels_aligment"
    experiment_folder.mkdir(exist_ok=True, parents=True)
    spec1, spec_len1, text1, text_len1, alignment1, cls_idx1, class_peak_value1 = load_data(i=0, target_cls=0)
    spec1, spec_len1 = spec1.unsqueeze(0), spec_len1.unsqueeze(0)
    spec1, spec_len1 = tp.model.decoder.squeezer.trim_dims(spec1, lens=spec_len1)

    context1, spec_len1 = get_combined_context_from_spec(spec1, spec_len1, text1, text_len1, alignment1)
    z1 = get_Z_from_spec(spec1.cuda(), spec_len1, context1)

    z = z1.clone()
    for ch in range(5):
        z[0, ch, :] *= 10
        pred_spec = predict_spec(z=z, spec_len=spec_len1, context=context1)
        sample_name = experiment_folder / f"sample_ch_{ch}*10.npz"
        np_spec = pred_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)
        z = z1.clone()

        z[0, ch, :] += 0.3
        pred_spec = predict_spec(z=z, spec_len=spec_len1, context=context1)
        sample_name = experiment_folder / f"sample_ch_{ch}+0,3.npz"
        np_spec = pred_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)
        z = z1.clone()

        z[0, ch, :z.shape[2]//4] *= 10
        pred_spec = predict_spec(z=z, spec_len=spec_len1, context=context1)
        sample_name = experiment_folder / f"sample_ch_{ch}*10_quoter_ch.npz"
        np_spec = pred_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)
        z = z1.clone()

        z[0, ch, 3*z.shape[2]//4:] *= 10
        pred_spec = predict_spec(z=z, spec_len=spec_len1, context=context1)
        sample_name = experiment_folder / f"sample_ch_{ch}*10_last_quoter_ch.npz"
        np_spec = pred_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)
        z = z1.clone()

        z[0, ch, :] = 0.3
        pred_spec = predict_spec(z=z, spec_len=spec_len1, context=context1)
        sample_name = experiment_folder / f"sample_ch_{ch}=0,3.npz"
        np_spec = pred_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)
        z = z1.clone()

        z[0, ch, :] = 0.0
        pred_spec = predict_spec(z=z, spec_len=spec_len1, context=context1)
        sample_name = experiment_folder / f"sample_ch_{ch}=0,0.npz"
        np_spec = pred_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)
        z = z1.clone()
    return experiment_folder

# experiment_folder = change_Z_channels()
# run_audio_pred(data_path=experiment_folder)

def change_context_channels():
    pass

def inference_distribution_test():
    # inference often differs with std = 1, gets significantly closer with std = 0 # mb test on mse or sth
    # conv class a bit higher mu, and much higher std in both targets and inference
    for i in range(5):
        for j in range(3):
            for t_cls in range(3):
                # get target S0
                spec, spec_len, text, text_len, alignment, cls_idx, class_peak_value = load_data(i=i, target_cls=t_cls)
                spec = spec.unsqueeze(0)

                cls_idx = torch.LongTensor([cls_idx])
                class_peak_value = torch.LongTensor([class_peak_value])

                # text_mask = training_utils.get_mask_from_lens(text_len)
                # inference spectrogram S1

                context1 = inference_phoneme_context(text, text_len, alignment)
                context1 = inference_combine_prior_context(context1, spec_len, class_peak_value, std=1)
                context1, spec_len = tp.model.decoder.squeezer.trim_dims(context1, lens=spec_len)

                spec, spec_len = tp.model.decoder.squeezer.trim_dims(spec, lens=spec_len)
                z1 = inference_z(text, spec_len).cuda()
                spec1 = predict_spec(z1, spec_len, context1)
                # check if VAE(S1) ~ N(cls, 1)
                z1, mu1, logs1 = tp.model.VAE(spec1)
                # inferencja z oryginalym contextem vs oryginal oba przez VAE
                z, mu, logs = tp.model.VAE(spec.cuda())

                print(f"t_cls {t_cls},target mu {mu.mean()} std {mu.std()}, mu predicted by VAE from inference spec {mu1.mean()} std {mu1.std()}, i {i}, j {j}")

inference_distribution_test()

def missmatched_aligment():
    # when Z and context come from different samples and same class it work without any loss of quality (which means that context works with different classes?)
    # with different classes only style of speech seems to change. It indicates that there is no phonetic information along time dimension?

    experiment_folder = experiments_folder / f"{folder_name_filter}missmatched_aligment"
    experiment_folder.mkdir(exist_ok=True, parents=True)
    spec1, spec_len1, text1, text_len1, alignment1, cls_idx1, class_peak_value1 = load_data(i=0, target_cls=0)
    spec1, spec_len1 = spec1.unsqueeze(0), spec_len1.unsqueeze(0)
    cls_idx1 = torch.LongTensor([cls_idx1])
    class_peak_value1 = torch.LongTensor([class_peak_value1])

    spec2, spec_len2, text2, text_len2, alignment2, cls_idx2, class_peak_value2 = load_data(i=1, target_cls=2)
    spec2, spec_len2 = spec2.unsqueeze(0), spec_len2.unsqueeze(0)
    cls_idx2 = torch.LongTensor([cls_idx2])
    class_peak_value2 = torch.LongTensor([class_peak_value2])

    # Z -> alignemnt2 * X = len1
    # spec1, spec_len1 = tp.model.decoder.squeezer.trim_dims(spec1, lens=spec_len)
    context2, spec_len2 = get_combined_context_from_spec(spec2, spec_len2, text2, text_len2, alignment2)
    context2, spec_len2 = tp.model.decoder.squeezer.trim_dims(context2, lens=spec_len2)
    spec2, spec_len2 = tp.model.decoder.squeezer.trim_dims(spec2, lens=spec_len2)
    z2 = get_Z_from_spec(spec2.cuda(), spec_len2, context2)

    # simpler version just overwrite alignment and cut to the same length
    # shorter_txt_len = min(alignment1.shape[2], alignment2.shape[2])
    # shorter_spec_len = min(alignment1.shape[3], alignment2.shape[3])
    # alignment1 = alignment1[..., :shorter_txt_len, :shorter_spec_len]
    # alignment2 = alignment2[..., :shorter_txt_len, :shorter_spec_len]
    # text2 = text2[:, :, :shorter_txt_len]
    # spec_len = torch.LongTensor([shorter_spec_len])

    # context -> aligment1 = len1
    context1, spec_len1 = get_combined_context_from_spec(spec1, spec_len1, text1, text_len1, alignment1)
    context1, spec_len1 = tp.model.decoder.squeezer.trim_dims(context1, lens=spec_len1)
    z1 = get_Z_from_spec(spec1.cuda(), spec_len1, context1)

    # spec = pred_spec(Z2, context1)
    context1 = context1[:z2.shape[2], ...]
    # context from utterance and z from different
    pred_spec = predict_spec(z2, spec_len2, context1)
    sample_name = experiment_folder / f"sample_z2_context1.npz"
    np_spec = pred_spec[0].cpu().data.numpy()
    save_spec(spec=np_spec, path=sample_name)

    # spec = pred_spec(Z1, context2)
    z1 = z1[:z2.shape[2]]
    # context from utterance and z from different
    pred_spec = predict_spec(z1, spec_len2, context2)
    sample_name = experiment_folder / f"sample_z1_context2.npz"
    np_spec = pred_spec[0].cpu().data.numpy()
    save_spec(spec=np_spec, path=sample_name)

    # sample 1
    sample_name = experiment_folder / f"sample1.npz"
    np_spec = spec1[0].cpu().data.numpy()
    save_spec(spec=np_spec, path=sample_name)

    # sample 2
    sample_name = experiment_folder / f"sample2.npz"
    np_spec = spec2[0].cpu().data.numpy()
    save_spec(spec=np_spec, path=sample_name)

    return experiment_folder

# experiment_folder = missmatched_aligment()
# run_audio_pred(data_path=experiment_folder)

# def measure_similarity():
#     # pass 100 spectrograms through VAE and see the distribution / mean or sth
#     # cosine distance between this and inference class context
#     # test if the come from the same distribution
#     pass


def compare_class_context_similarity():
    # should I compare VAE output or phoneme + VAE (i guess the 1st would be better)
    # phoneme + VAE cos sim does not indicate that predicted context of the same class are more similar to target than different class

    # compare similarity between different samples (are same class emb much similar even between samples)
    # does similarity over channels makes sense (with same content i guess)
    # similarity over time 1 | similarity over channels 2
    # similarity over time would correspond to phonemes so when alignments match it seems nice
    peak_cls_mapping = {0: 0, 1: 1, 2: -1}
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim_mu = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for target_cls in range(3):
        i = 0
        spec, spec_len, text, text_len, alignment, cls_idx, class_peak_value = load_data(i=i, target_cls=target_cls)
        spec = spec.unsqueeze(0)
        target_context, spec_len = get_combined_context_from_spec(spec, spec_len, text, text_len, alignment)
        target_z, target_mu, target_logs = tp.model.VAE(spec.cuda())
        phon_context = inference_phoneme_context(text, text_len, alignment)
        self_target_similarity = cos_sim(target_context, target_context).sum()
        print(f"self_target_similarity_cls{target_cls} {self_target_similarity}")
        print(f"target mu {target_mu.mean()}, logs {target_logs.mean()}")
        for pred_cls in range(3):
            mu = [peak_cls_mapping[pred_cls]]
            cls_peak = torch.LongTensor(mu)
            pred_context = inference_combine_prior_context(phon_context, spec_len, cls_peak=cls_peak, std=1)
            z = inference_z(text, spec_len).cuda()
            pred_spec = predict_spec(z=z, spec_len=spec_len, context=pred_context)
            pred_z, pred_mu, pred_logs = tp.model.VAE(pred_spec.cuda())
            target_similarity = cos_sim(target_context, pred_context).sum()
            print(f"cos_sim(target_context_cls{target_cls}, pred_context_cls{pred_cls}) {target_similarity}")
            mu_sim = cos_sim_mu(target_mu, pred_mu)
            print(f"pred mu {pred_mu.mean()}, logs {pred_logs.mean()}")
            print(f"mu similarity {mu_sim.mean()}")
            for pred2_cls in range(3):
                mu = [peak_cls_mapping[pred_cls]]
                cls_peak = torch.LongTensor(mu)
                pred_context2 = inference_combine_prior_context(phon_context, spec_len, cls_peak=cls_peak, std=1)
                target_similarity = cos_sim(pred_context, pred_context2).sum()
                print(f"cos_sim(pred_context_cls{pred_cls}, pred_context_cls{pred2_cls}) {target_similarity}")

def interpolate():
    # create interpolations Z1 Z2 (try in spherical coordinates)
    # generate samples with context1, context2, interpolate context???
    # clip Z1 in T dim to be same len as Z2

    # do the interpolation for inference and check if effect is the same

    experiment_folder = experiments_folder / f"{folder_name_filter}interpolate"
    experiment_folder.mkdir(exist_ok=True, parents=True)
    spec1, spec_len1, text1, text_len1, alignment1, cls_idx1, class_peak_value1 = load_data(i=0, target_cls=0)
    spec1, spec_len1 = spec1.unsqueeze(0), spec_len1.unsqueeze(0)
    cls_idx1 = torch.LongTensor([cls_idx1])
    class_peak_value1 = torch.LongTensor([class_peak_value1])

    spec2, spec_len2, text2, text_len2, alignment2, cls_idx2, class_peak_value2 = load_data(i=1, target_cls=2)
    spec2, spec_len2 = spec2.unsqueeze(0), spec_len2.unsqueeze(0)
    cls_idx2 = torch.LongTensor([cls_idx2])
    class_peak_value2 = torch.LongTensor([class_peak_value2])

    context1, spec_len1 = get_combined_context_from_spec(spec1, spec_len1, text1, text_len1, alignment1)
    context1, spec_len1 = tp.model.decoder.squeezer.trim_dims(context1, lens=spec_len1)
    z1 = get_Z_from_spec(spec1.cuda(), spec_len1, context1)

    context2, spec_len2 = get_combined_context_from_spec(spec2, spec_len2, text2, text_len2, alignment2)
    context2, spec_len2 = tp.model.decoder.squeezer.trim_dims(context2, lens=spec_len2)
    spec2, spec_len2 = tp.model.decoder.squeezer.trim_dims(spec2, lens=spec_len2)
    z2 = get_Z_from_spec(spec2.cuda(), spec_len2, context2)

    if spec_len1[0] > spec_len2[0]:
        spec_len = spec_len2
    else:
        spec_len = spec_len1

    context1 = context1[:z2.shape[2], ...]
    context2 = context1[:z1.shape[2], ...]
    z1 = z1[..., :z2.shape[2]]
    z2 = z2[..., :z1.shape[2]]

    for i in range(6):
        i = i / 5
        prop1 = (1 - i)
        prop2 = i
        z = z1 * prop1 + z2 * prop2
        pred_z_spec = predict_spec(z, spec_len, context1)
        sample_name = experiment_folder / f"sample_{prop1}_z1_{prop2}_z2_context1.npz"
        np_spec = pred_z_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)

        context = context1 * prop1 + context2 * prop2
        pred_cont_spec = predict_spec(z1, spec_len, context)
        sample_name = experiment_folder / f"sample_{prop1}_context1_{prop2}_context2_z1.npz"
        np_spec = pred_cont_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)

    vae_context1, mu, logs = tp.model.VAE(spec1.cuda())
    vae_context2, mu, logs = tp.model.VAE(spec2.cuda())

    for i in range(6):
        i = i / 5
        prop1 = (1 - i)
        prop2 = i
        text_mask = training_utils.get_mask_from_lens(text_len1)
        text_encodings = tp.model.text_encoder(text1, text_mask)
        context_attn, _, spec_len = tp.model.get_context(alignment=alignment1, text=text1, text_encodings=text_encodings,
                                                         text_mask=text_mask, spec_len=spec_len)

        vae_context = vae_context1 * prop1 + vae_context2 * prop2
        context = tp.model.context_combiner(attn=context_attn, aux_context=vae_context)

        context = context[:z1.shape[2], ...]
        # context = context1 * prop1 + context2 * prop2
        pred_cont_spec = predict_spec(z1, spec_len2, context)
        sample_name = experiment_folder / f"sample_{prop1}_vae_context1_{prop2}_vae_context2_z1.npz"
        np_spec = pred_cont_spec[0].cpu().data.numpy()
        save_spec(spec=np_spec, path=sample_name)

    sample_name = experiment_folder / f"target1.npz"
    np_spec = spec1[0].cpu().data.numpy()
    save_spec(spec=np_spec, path=sample_name)

    sample_name = experiment_folder / f"target2.npz"
    np_spec = spec2[0].cpu().data.numpy()
    save_spec(spec=np_spec, path=sample_name)

    return experiment_folder

def phoneme_Z_properties():
    # phonemes have significantly different mean from global mean of spectrogram
    # property isn't there during sampling cuz i just sample from N(c, 1)
    for i in range(3):
        spec, spec_len, text, text_len, alignment, cls_idx, class_peak_value = load_data(i=0, target_cls=i)
        cls_peak = torch.LongTensor([class_peak_value])
        spec = spec.unsqueeze(0)
        # z1 = inference_z(text, spec_len)
        # context1 = inference_phoneme_context(text, text_len, alignment)
        # context1 = inference_combine_prior_context(context1, spec_len, class_peak_value, std=1)
        # spec1 = predict_spec(z1, spec_len, context1)

        # split per phoneme and see if mean is diff and if similar between (does it make sense -> check shape of mu to make sure)
        context1, spec_len1 = get_combined_context_from_spec(spec, spec_len, text, text_len, alignment)
        context1, spec_len1 = tp.model.decoder.squeezer.trim_dims(context1, lens=spec_len1)
        z = get_Z_from_spec(spec.cuda(), spec_len1, context1)
        alignment = alignment[..., :z.shape[2]]

        pred_z = inference_z(text, spec_len).cuda()
        pred_z = pred_z[..., :z.shape[2]]

        print(f"cls {i}, z mean {z.mean()}, std {z.std()}, pred z {pred_z.mean()}, std {pred_z.std()}")
        for i in range(5): # alignment.shape[2]//5
            norm = alignment[0, 0, i, :].sum().item() * 80
            z_phoneme_mean = (alignment[0, 0, i, :] * z).sum() / norm
            z_pred_phoneme_mean = (alignment[0, 0, i, :] * pred_z).sum() / norm
            print(f"phoneme {i} mean target {z_phoneme_mean}, pred {z_pred_phoneme_mean}")


# czy VAE mocniej kontroluje styl niz class embedding / pokazac ze Z jest class independent
# biore Z z netural ( i Z zsamplowany czy pozostala informacja w Z ze spectrogramu dominuje styl z contextu)
# centroid per klasa na outputcie VAE / inferencja z innym stylem przez samplowanie
# benchmark vs class embedding

# Z_neu -> Z_conv i reszta tak samo czy transferuje sie styl (interpolacja)

# oczekiwane wyniki
# Z zawiera informacje o stylu
# Z ze spectogramu zawadza w przekazwywaniu stylu przez context
# przy inferencji Z przestaje dominowac i context zaczyna efektywnie przekazywac styl
# VAE lepiej przekazuje styl niz embedding
