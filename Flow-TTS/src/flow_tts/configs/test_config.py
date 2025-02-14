type = "channel_dependant"
sub_dist_count = 4

logging_interval = 1
checkpoint_path = "checkpoint_model.pth"
ds_type = "test"
mean = [-10, -5, 5, 10]
std = [1, 1, 1, 1]
sample_shapes = (1, 80, 80)
sample_count = 100

model_dir = f"logs/REMOVE_{ds_type}_{type}_ncop_means_{'_'.join(str(k) for k in mean)}_stds_{'_'.join(str(k) for k in std)}_1sqz"
