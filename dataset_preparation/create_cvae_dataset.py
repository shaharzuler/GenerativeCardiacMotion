import os

import numpy as np
from scipy.ndimage import zoom

from img_features import extract_img_features

data_source = "/home/shahar/projects/CVAE_proj/CVAE/data/data_source"
all_paths = sorted(os.listdir(data_source))

target_dir = "/home/shahar/projects/CVAE_proj/CVAE/data/data_for_cvae"

img_paths  = [path for path in all_paths if "template_img" in path]
flow_paths = [path for path in all_paths if "flow" in path]

img_paths = [os.path.join(data_source, f) for f in img_paths]
flow_paths = [os.path.join(data_source, f) for f in flow_paths]

flow_target_dir = f"{target_dir}/flows"
img_features_target_dir = f"{target_dir}/img_features"

for n_sample, (img_path, flow_path) in enumerate(zip(img_paths, flow_paths)):
    conditions_pyramid, data = extract_img_features(img_path, flow_path)
    flow = data["flows_gt"]
    if flow.shape != (1,3,86,86,76) and max(abs(np.array(flow.shape)-np.array((1,3,86,86,76))))==1:
        flow = zoom(flow, (1,1,86/flow.shape[2],86/flow.shape[3],76/flow.shape[4]))
    np.save(f"{flow_target_dir}/flow_sample_{n_sample}.npy", np.array(flow)[0])
    for n_layer, layer in enumerate(conditions_pyramid):
        np.save(f"{img_features_target_dir}/img_features_layer_{n_layer}_sample_{n_sample}.npy", layer.detach().cpu().numpy()[0])


