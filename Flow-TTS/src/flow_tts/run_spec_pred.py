from pipeline import pipeline

data_path = "/home/ec2-user/renard/preds/all_dcVAE_new12block_samples_centroid"
txt_file = "prompts_{mode}.txt"
checkpoint_path = "/home/ec2-user/renard/Flow-TTS/src/flow_tts/logs/all_dcVAE_new12block/checkpoint_model.pth"
samples_count = 4
modes = ['train', 'eval', 'inference']
stds = [0.1, 0.2]

overwrite_style = True
sample_classes = [0, 1, 2]
styles = [(0, 0.0), (1, 1.0), (2, -1.0)]

inf_pipeline = pipeline.VAEInferencePipeline.from_checkpoint(
    checkpoint_path=checkpoint_path,
    samples_count=samples_count,
    data_path=data_path,
    txt_file=txt_file,
    modes=modes,
    stds=stds,
    overwrite_style=overwrite_style,
    sample_classes=sample_classes,
    styles=styles,
    rank=0,
)
inf_pipeline.inference_known_align()

