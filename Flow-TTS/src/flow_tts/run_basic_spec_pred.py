from pipeline import pipeline

data_path = "/home/ec2-user/renard/data/preds/VAE_all_dcVAE_nl_cs_bic_hui_peaks1"
txt_file = "prompts_{mode}.txt"
checkpoint_path = "/home/ec2-user/renard/Flow-TTS/src/flow_tts/logs/VAE_all_dcVAE_nl_cs_bic_hui_peaks1/checkpoint_model.pth"
samples_count = 10
modes = ['train', 'eval']

inf_pipeline = pipeline.InferencePipeline.from_checkpoint(
    checkpoint_path=checkpoint_path,
    samples_count=samples_count,
    data_path=data_path,
    txt_file=txt_file,
    modes=modes,
    stds=None,
    overwrite_style=True,
    sample_classes=None,
    styles=None,
    rank=0,
)
inf_pipeline.inference_known_align()
