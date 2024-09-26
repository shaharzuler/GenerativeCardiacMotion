import os
import time
import json

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
from flow_n_corr_utils import disp_flow_colors

from model import CVAE
from dataset import cvae_dataset
torch.manual_seed = 21


def infer_cvae(l1o_idx, top_out_path, device, config):
    torch.cuda.set_device(device) 
    DEVICE = torch.device(device)
    out_path = f"{top_out_path}/{l1o_idx}"
    print(out_path)
    os.makedirs(out_path)
    os.makedirs(out_path+"/infer/random")
    os.makedirs(out_path+"/infer/2dims_and0")
    os.makedirs(out_path+"/infer/2dims_and0_grid")

    val_set = cvae_dataset("/home/shahar/projects/CVAE_proj/CVAE/data/data_for_cvae", device=DEVICE, l1o_idx=l1o_idx, train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    num_chs = [3, 16, 32, 64, 96, 128, 128] 
    cvae_model = CVAE(num_chs=num_chs, max_flow_hat_abs_val=50).to(DEVICE)
    model_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240913_215544/{l1o_idx}/model_final.pt"
    cvae_model.load_state_dict(torch.load(model_path))
    cvae_model.eval()
    
    for flow, conditions_pyramid in val_loader: 
        num_generations = 30
        np.save(f"{out_path}/infer/random/condition.npy", conditions_pyramid[-1].detach().cpu().numpy())
        plt.imshow(conditions_pyramid[-1].detach().cpu().numpy()[0,0,43], cmap="bone")
        plt.savefig(f"{out_path}/infer/random/condition.png")

        for gen in range(num_generations):
            flow_hat_generated, z = cvae_model.generate(conditions_pyramid, device=DEVICE, output_z=True, provide_z=False)
            flow_hat_generated = flow_hat_generated.detach().cpu().numpy()
            z = z.detach().cpu().numpy()

            np.save(f"{out_path}/infer/random/flow_generated_{gen}.npy", flow_hat_generated)
            np.save(f"{out_path}/infer/random/z_{gen}.npy", z)
            viss=disp_flow_colors(flow_hat_generated[0])
            for dim, section in zip(("x","y","z"),viss):
                plt.imshow(section.transpose())
                plt.savefig(f"{out_path}/infer/random/flow_section_{dim}_{gen}")

        np.save(f"{out_path}/infer/2dims_and0_grid/condition.npy", conditions_pyramid[-1].detach().cpu().numpy())
        plt.imshow(conditions_pyramid[-1].detach().cpu().numpy()[0,0,43], cmap="bone")
        plt.savefig(f"{out_path}/infer/2dims_and0_grid/condition.png")
        for x in np.linspace(-1,1,10):
            for y in np.linspace(-1,1,10):
                z_size = (1,128,3,3,3)
                z = torch.zeros(z_size).to(DEVICE0)
                z[0,:60,:,:,:] = x
                z[0,61:,:,:,:] = y
                flow_hat_generated, z = cvae_model.generate(conditions_pyramid, device=DEVICE, output_z=True, provide_z=z)
                flow_hat_generated = flow_hat_generated.detach().cpu().numpy()
                z = z.detach().cpu().numpy()

                np.save(f"{out_path}/infer/2dims_and0_grid/flow_generated_x_{x:.3f}_y_{y:.3f}.npy", flow_hat_generated)
                np.save(f"{out_path}/infer/2dims_and0_grid/x_{x:.3f}_y_{y:.3f}.npy", z)
                viss=disp_flow_colors(flow_hat_generated[0])
                for dim, section in zip(("x","y","z"),viss):
                    plt.imshow(section.transpose())
                    plt.savefig(f"{out_path}/infer/2dims_and0_grid/flow_section_{dim}_x_{x:.3f}_y_{y:.3f}.png")

        np.save(f"{out_path}/infer/2dims_and0/condition.npy", conditions_pyramid[-1].detach().cpu().numpy())
        plt.imshow(conditions_pyramid[-1].detach().cpu().numpy()[0,0,43], cmap="bone")
        plt.savefig(f"{out_path}/infer/2dims_and0/condition.png")
        for gen in range(num_generations):
            z_size = (1,128,3,3,3)
            z = torch.zeros(z_size).to(DEVICE0)
            z[0,:60,:,:,:] = np.random.normal()
            z[0,:61,:,:,:] = np.random.normal()
            flow_hat_generated, z = cvae_model.generate(conditions_pyramid, device=DEVICE, output_z=True, provide_z=z)
            flow_hat_generated = flow_hat_generated.detach().cpu().numpy()
            z = z.detach().cpu().numpy()

            np.save(f"{out_path}/infer/2dims_and0/flow_generated_{gen}.npy", flow_hat_generated)
            np.save(f"{out_path}/infer/2dims_and0/z_{gen}.npy", z)
            viss=disp_flow_colors(flow_hat_generated[0])
            for dim, section in zip(("x","y","z"),viss):
                plt.imshow(section.transpose())
                plt.savefig(f"{out_path}/infer/2dims_and0/flow_section_{dim}_{gen}.png")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    DEVICE0 = "cuda:0"
    DEVICE1 = "cuda:1"

    top_out_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(top_out_path)
    
    config_file_path = "/home/shahar/projects/CVAE_proj/CVAE/config.json"
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    output_config_path = os.path.join(top_out_path, "config.json")
    with open(output_config_path, 'w') as output_config_file:
        json.dump(config, output_config_file, indent=4)

    l1o_idx, device = 2, DEVICE0
    infer_cvae(l1o_idx, top_out_path, device, config)       
    
    print("Finished")

