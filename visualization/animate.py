import os

from PIL import Image
import numpy as np
import torch
from torch import nn
import imageio
from matplotlib import pyplot as plt

from flow_n_corr_utils import warp, disp_flow_colors

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
    image_warped = nn.functional.grid_sample(image, v_grid, align_corners=False, padding_mode="border", mode="bilinear")

    return image_warped[0,0,:,:,:].cpu().numpy()

def save_img_pair(img, flow, out_path, idx, suffix=""):
    shape_ = img.shape[2:]
    orig_img = warp(img, torch.zeros_like(torch.tensor(flow)))
    warped_img = warp(img, flow)
    orig_frame = np.concatenate([ orig_img[shape_[0]//2, :, :], orig_img[:, shape_[1]//2, :], orig_img[:, :, shape_[2]//2] ], axis=1) * 255
    warped_frame = np.concatenate([ warped_img[shape_[0]//2, :, :], warped_img[:, shape_[1]//2, :], warped_img[:, :, shape_[2]//2] ], axis=1) * 255

    orig_frame = np.uint8(orig_frame)
    warped_frame = np.uint8(warped_frame)

    imageio.imwrite(f"{out_path}/generated_orig_systole{idx}{suffix}.png", orig_frame)
    imageio.imwrite(f"{out_path}/generated_warped_diastole{idx}{suffix}.png", warped_frame)
    viss=disp_flow_colors(flow[0])
    for dim, section in zip(("x","y","z"),viss):
        plt.imshow(section.transpose())
        plt.savefig(f"{out_path}/flow_colors_{dim}_{idx}{suffix}.png")

def animate(img, flow, out_path, idx, suffix="", num_timesteps=10):
    shape_ = img.shape[2:]
    half_seq = []#[img[0,0]]
    for timestep in range((num_timesteps//2)+1):
        flow_ts = flow * (timestep)/(num_timesteps//2)
        warped_img = warp(img, flow_ts)
        half_seq.append(warped_img)
    seq = half_seq + half_seq[::-1][1:-1]
    # np.save(f"{out_path}/seq_arr{idx}.npy", np.array(seq))
    frames = []
    for ts in seq:
        frame = np.concatenate([ ts[shape_[0]//2, :, :],ts[:, shape_[1]//2, :], ts[:, :, shape_[2]//2] ], axis=1) * 255
        frames.append(Image.fromarray(frame ) )
    frames[0].save(f"{out_path}/generated{idx}{suffix}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

gen_1_gif_per_val_sample = False
if gen_1_gif_per_val_sample:
    for idx in range(1,56):
        img_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/{idx}/condition.npy"
        flow_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/{idx}/flow_generated_final.npy"
        if os.path.isfile(img_path) and os.path.isfile(flow_path):
            img = np.load(img_path)
            flow = np.load(flow_path)
            animate(img, flow, "/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/vis_results", idx)
            #GT:
            flow = np.expand_dims(np.load(f"/home/shahar/projects/CVAE_proj/CVAE/data/data_for_cvae/flows/flow_sample_{idx}.npy"), 0)
            animate(img, flow, "/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/vis_results", idx, suffix="_gt")

gen_images_per_val_sample = False
if gen_images_per_val_sample:
    for idx in range(1,56):
        img_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/{idx}/condition.npy"
        flow_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/{idx}/flow_generated_final.npy"
        if os.path.isfile(img_path) and os.path.isfile(flow_path):
            img = np.load(img_path)
            flow = np.load(flow_path)
            save_img_pair(img, flow, "/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/vis_results", idx)
            #gt:
            flow = np.expand_dims(np.load(f"/home/shahar/projects/CVAE_proj/CVAE/data/data_for_cvae/flows/flow_sample_{idx}.npy"), 0)
            save_img_pair(img, flow, "/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/vis_results", idx, suffix="_gt")

gen_many_gifs_for_large_grid_gif = False
if gen_many_gifs_for_large_grid_gif:
    imgs_dir = "/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240922_170624/2/infer/2dims_and0_grid"
    img_path=os.path.join(imgs_dir,"condition.npy")
    for filename in os.listdir(imgs_dir):
        if f"flow_generated_x_" in filename and ".npy" in filename:
            flow_path = os.path.join(imgs_dir, filename)
            if os.path.isfile(img_path) and os.path.isfile(flow_path):
                img = np.load(img_path)
                flow = np.load(flow_path)
                animate(img, flow, os.path.join(imgs_dir,"vis_results"), "_"+filename.replace("flow_generated_", "").replace(".npy", ""))
