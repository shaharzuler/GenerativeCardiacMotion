from PIL import Image
import numpy as np
import torch
from torch import nn

from flow_n_corr_utils import warp

def _mesh_grid(B:int, H:int, W:int, D:int)->np.array:
    # batches not implented
    x = torch.arange(H)
    y = torch.arange(W)
    z = torch.arange(D)
    mesh = torch.stack(torch.meshgrid(x, y, z)[::-1], 0) 

    mesh = mesh.unsqueeze(0)
    return mesh.repeat([B,1,1,1,1])

def _norm_grid(v_grid:np.array)->np.array:
    """scale grid to [-1,1]"""
    _, _, H, W, D = v_grid.size()
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :, :] = (2.0 * v_grid[:, 0, :, :, :] / (D - 1)) - 1.0 
    v_grid_norm[:, 1, :, :, :] = (2.0 * v_grid[:, 1, :, :, :] / (W - 1)) - 1.0
    v_grid_norm[:, 2, :, :, :] = (2.0 * v_grid[:, 2, :, :, :] / (H - 1)) - 1.0 
    
    return v_grid_norm.permute(0, 2, 3, 4, 1)

def warp(image, flow):
    flow = torch.tensor(flow)
    image = torch.tensor(image)
    B, _, H, W, D = flow.size()
    flow = torch.flip(flow, [1]) # flow is now z, y, x
    base_grid = _mesh_grid(B, H, W, D).type_as(image)  # B2HW
    grid_plus_flow = base_grid + flow
    v_grid = _norm_grid(grid_plus_flow)  # BHW2
    image_warped = nn.functional.grid_sample(image, v_grid, align_corners=False, padding_mode="border", mode="nearest")

    return image_warped[0,0,:,:,:].cpu().numpy()






def animate(img, flow, out_path, idx, num_timesteps=10):
    scale_down_flow_by = 10
    flow /= scale_down_flow_by
    shape_ = img.shape[2:]
    half_seq = [img[0,0]]
    for timestep in range(num_timesteps//2):
        flow_ts = flow * (timestep+1)*(num_timesteps//2)
        warped_img = warp(img, flow_ts)
        half_seq.append(warped_img)
    seq = half_seq + half_seq[::-1][1:-1]
    # np.save(f"{out_path}/seq_arr{idx}.npy", np.array(seq))
    frames = []
    for ts in seq:
        frame = np.concatenate([ ts[shape_[0]//2, :, :],ts[:, shape_[1]//2, :], ts[:, :, shape_[2]//2] ], axis=1) * 255
        frames.append(Image.fromarray(frame ) )
    frames[0].save(f"{out_path}/generated{idx}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)


for idx in range(56):
    img_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240911_105603/{idx}/condition.npy"
    flow_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240911_105603/{idx}/flow_generated_final.npy"
    import os
    if os.path.isfile(img_path) and os.path.isfile(flow_path):
        img = np.load(img_path)
        flow = np.load(flow_path)
        animate(img, flow, "/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240911_105603/vis_results", idx)

    
    