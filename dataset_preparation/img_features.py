from easydict import EasyDict

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
        template_image_path=img_path,
        unlabeled_image_path=img_path,
        template_LV_seg_path=None,
        unlabeled_LV_seg_path=None,
        template_shell_seg_path=None,
        unlabeled_shell_seg_path=None,
        flows_gt_path=flow_path,
        args=EasyDict(args),
        feature_extractor_mode=True,
    )
    return x1_p, data