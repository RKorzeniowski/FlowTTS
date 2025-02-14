# Flow-TTS

** Describe Flow-TTS **

## Training

### FlowTTS

#### Run training

In file `run_training.py` import your config and set it as `config` variable.

Important config variables that you want to overwrite when using default `custom_config.py` config:
- `model_dir` - place where model checkpoint, logs, hparams and config used for training are saved.
- `root_dir` - path to folder with alignment, prompts, melspec folder and phonemes folder.
- `class` - prefix present in names of training npz spectrogram files.
- `mel_folder` - name of folder with melspectrograms.
- `phonemes_folder` - path with phoneme to melspec alignments.
- `vocab_file` - index to phoneme mapping.

Imporant hparams variables:
- `dec_num_blocks` - number of blocks in a decoder 
- `num_dec_block_layers` - number of flow layers in each decoder block

Then run command `python run_training.py`.

## Generate Samples

#### Run melspetrogram generation

Set `data_path` - where synthesized melspectrogram will show up and `checkpoint_path` - path to saved model.

Run command `python run_basic_spec_pred.py`.

#### Run audio generation

Run open source Vocoder on synthesized melspectrograms. Remember to adjust melspectrograms based on the format expected
by the Vocoder. 

