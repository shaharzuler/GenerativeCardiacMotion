from easydict import EasyDict

import torch
import numpy as np

from four_d_ct_cost_unrolling import  infer_backbone
from four_d_ct_cost_unrolling import get_default_backbone_config, get_default_checkpoints_path


def extract_img_features(img_path, flow_path, req_shape=(86,86,76)):
    args = get_default_backbone_config()
    args["inference_args"]["inference_flow_median_filter_size"] = False
    args["visualization_arrow_scale_factor"] = 1
    args["cuda_device"] = 0
    args["scale_down_by"] = 2
    args["load"] = get_default_checkpoints_path()

    current_shape = np.load(img_path).shape
    args["scale_down_by"] = np.array(current_shape)/np.array(req_shape)

    x1_p, data = infer_backbone(
        template_image_path=img_path,#'/home/fiman/projects/DL_course/spectral_diffusion/spectral_diffusion/cardiac_sample_data/cardiac_sample_data/cardiac_sample_data/image_skewed_thetas_100.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
        unlabeled_image_path=img_path,#'/home/shahar/projects/CVAE_proj/CVAE/data_source/image_skewed_thetas_82.5_-27.5_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
        template_LV_seg_path=img_path.replace("_img_","_mask_"),#'/home/shahar/projects/CVAE_proj/CVAE/data_source/image_skewed_thetas_82.5_-27.5_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
        unlabeled_LV_seg_path=img_path.replace("_img_","_mask_"),#'/home/shahar/projects/CVAE_proj/CVAE/data_source/image_skewed_thetas_82.5_-27.5_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
        template_shell_seg_path=img_path.replace("_img_","_mask_"),#'/home/shahar/projects/CVAE_proj/CVAE/data_source/image_skewed_thetas_82.5_-27.5_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
        unlabeled_shell_seg_path=img_path.replace("_img_","_mask_"),#'/home/shahar/projects/CVAE_proj/CVAE/data_source/image_skewed_thetas_82.5_-27.5_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy',
        flows_gt_path=flow_path,
        args=EasyDict(args),
        feature_extractor_mode=True,
    )
    return x1_p, data