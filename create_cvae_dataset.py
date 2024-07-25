import numpy as np
import torch
import os

from img_features import extract_img_features
data_source = "/home/shahar/projects/CVAE_proj/CVAE/data_source"
all_paths = sorted(os.listdir(data_source))

# img_path = '/home/fiman/projects/DL_course/spectral_diffusion/spectral_diffusion/cardiac_sample_data/cardiac_sample_data/cardiac_sample_data/image_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy'
# flow_path = '/home/fiman/projects/DL_course/spectral_diffusion/spectral_diffusion/cardiac_sample_data/cardiac_sample_data/cardiac_sample_data/flow_for_image_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy'
target_dir = "/home/shahar/projects/CVAE_proj/CVAE/data_for_cvae"

img_paths  = all_paths[len(all_paths)//2:]
flow_paths = all_paths[:len(all_paths)//2]

img_paths = [os.path.join(data_source, f) for f in img_paths]
flow_paths = [os.path.join(data_source, f) for f in flow_paths]

flow_target_dir = f"{target_dir}/flows"
img_features_target_dir = f"{target_dir}/img_features"

for n_sample, (img_path, flow_path) in enumerate(zip(img_paths, flow_paths)):
    conditions_pyramid, data = extract_img_features(img_path, flow_path)
    flow = data["flows_gt"]
    np.save(f"{flow_target_dir}/flow_sample_{n_sample}.npy", np.array(flow)[0])
    for n_layer, layer in enumerate(conditions_pyramid):
        np.save(f"{img_features_target_dir}/img_features_layer_{n_layer}_sample_{n_sample}.npy", layer.detach().cpu().numpy()[0])


