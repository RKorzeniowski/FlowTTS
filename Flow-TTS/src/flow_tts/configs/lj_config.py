logging_interval = 1
checkpoint_path = "checkpoint_model.pth"
model_dir = "logs/lj_clamp_30dec_melmaxpool"

mel_folder = "mf"
suffix = ".npz"
spec_transform = None

train_audiopaths_and_text = "/home/ubuntu/renard/Flow-TTS/src/flow_tts/filelists/ljs_audio_text_train_filelist.txt"
eval_audiopaths_and_text = "/home/ubuntu/renard/Flow-TTS/src/flow_tts/filelists/ljs_audio_text_val_filelist.txt"
text_cleaners = ["english_cleaners"]
cmudict_path = "./data/cmu_dictionary"

ds_type = "lj"
