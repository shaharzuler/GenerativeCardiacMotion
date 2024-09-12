import os
import time
import json

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from four_d_ct_cost_unrolling import torch_to_np
from flow_n_corr_utils import min_max_norm

from cvae_model import CVAE
from dataset import cvae_dataset
torch.manual_seed = 21





def train_cvae(l1o_idx, top_out_path, device, config):
    torch.cuda.set_device(device) 
    DEVICE = torch.device(device)
    out_path = f"{top_out_path}/{l1o_idx}"#f"/home/shahar/projects/CVAE_proj/CVAE/outputs/{time.strftime('%Y%m%d_%H%M%S')}"
    print(out_path)
    os.makedirs(out_path)
    os.makedirs(out_path+"/train")
    os.makedirs(out_path+"/eval")
    os.makedirs(out_path+"/z_generation")
    train_writer = SummaryWriter(out_path+"/train")
    eval_writer = SummaryWriter(out_path+"/eval")
    generation_writer = SummaryWriter(out_path+"/z_generation")
    train_set = cvae_dataset("/home/shahar/projects/CVAE_proj/CVAE/data_for_cvae", device=DEVICE, l1o_idx=l1o_idx, train=True, augmentations=True, augmentation_params=config["augmentation_params"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_set = cvae_dataset("/home/shahar/projects/CVAE_proj/CVAE/data_for_cvae", device=DEVICE, l1o_idx=l1o_idx, train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)


    num_chs = [3, 16, 32, 64, 96, 128, 128] # [3, 16, 32, 64, 96, 96] #
    cvae_model = CVAE(num_chs=num_chs, max_flow_hat_abs_val=50).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cvae_model.parameters(), lr=1e-3)

    lambda_reconstraction = config["lambda_reconstraction"]#0.85#0.7 #0.5 #0.1#0.05#0.5
    lambda_kl = config["lambda_kl"]#0.15#0.3 #0.5 #0.001#0.0001 nan

    num_epochs = config["num_epochs"]#1000
    for num_epoch in range(num_epochs):
        cvae_model.train()
        epoch_train_acc_loss = 0
        epoch_train_reconstraction_loss = 0
        epoch_train_kl = 0
        for batch_num, (flow, conditions_pyramid) in enumerate(train_loader): 
            optimizer.zero_grad()
            flow_hat, train_kl = cvae_model(flow, conditions_pyramid)
            train_reconstraction_loss = criterion(flow_hat, flow)
            train_loss = lambda_reconstraction * train_reconstraction_loss + lambda_kl * train_kl
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae_model.parameters(), max_norm=config["grad_max_norm"])
            optimizer.step()
            epoch_train_acc_loss += train_loss.item()
            epoch_train_kl += train_kl.item()
            epoch_train_reconstraction_loss += train_reconstraction_loss.item()
        epoch_train_acc_loss /= (batch_num+1)
        epoch_train_reconstraction_loss /= (batch_num+1)
        epoch_train_kl /= (batch_num+1)
        train_writer.add_scalar("reconstraction_loss", epoch_train_reconstraction_loss, num_epoch)
        train_writer.add_scalar("kl", epoch_train_kl, num_epoch)
        train_writer.add_scalar("total_loss", epoch_train_acc_loss, num_epoch)
        train_writer.add_image("f_hat",min_max_norm(flow_hat[0, :, :, 60, :]), num_epoch) # 3 h w
        train_writer.add_image("f",min_max_norm(flow[0, :, :, 60, :]), num_epoch) # 3 h w
        train_writer.add_image("img",min_max_norm(conditions_pyramid[-1][0, :, :, 60, :]), num_epoch) # 3 h w

        cvae_model.eval()
        epoch_eval_acc_loss = 0
        epoch_eval_reconstraction_loss = 0
        epoch_eval_generation_loss = 0
        epoch_eval_kl = 0
        for batch_num, (flow, conditions_pyramid) in enumerate(val_loader): 
            flow_hat, eval_kl = cvae_model(flow, conditions_pyramid)
            eval_reconstraction_loss = criterion(flow_hat, flow)
            eval_loss = lambda_reconstraction * eval_reconstraction_loss + lambda_kl * eval_kl
            epoch_eval_acc_loss += eval_loss.item()
            epoch_eval_kl += eval_kl.item()
            epoch_eval_reconstraction_loss += eval_reconstraction_loss.item()
            
            flow_hat_generated = cvae_model.generate(conditions_pyramid, device=DEVICE)
            eval_generation_loss = criterion(flow_hat_generated, flow)
            epoch_eval_generation_loss += eval_generation_loss.item()

        epoch_eval_acc_loss /= (batch_num+1)
        epoch_eval_reconstraction_loss /= (batch_num+1)
        epoch_eval_kl /= (batch_num+1)
        epoch_eval_generation_loss /= (batch_num+1)

        eval_writer.add_scalar("reconstraction_loss", epoch_eval_reconstraction_loss, num_epoch)
        eval_writer.add_scalar("kl", epoch_eval_kl, num_epoch)
        eval_writer.add_scalar("total_loss", epoch_eval_acc_loss, num_epoch)
        generation_writer.add_scalar("reconstraction_loss", epoch_eval_generation_loss, num_epoch)
        eval_writer.add_image("f_hat",min_max_norm(flow_hat[0, :, :, 60, :]), num_epoch) # 3 h w
        generation_writer.add_image("f_hat",min_max_norm(flow_hat_generated[0, :, :, 60, :]), num_epoch) # 3 h w
        eval_writer.add_image("f",min_max_norm(flow[0, :, :, 60, :]), num_epoch) # 3 h w
        eval_writer.add_image("img",min_max_norm(conditions_pyramid[-1][0, :, :, 60, :]), num_epoch) # 3 h w
        if num_epoch % config["save_weights_every"] == 0:
            torch.save(cvae_model.state_dict(), f"{out_path}/model_e{num_epoch}.pt")
            for batch_num, (flow, conditions_pyramid) in enumerate(val_loader):        
                flow_hat_generated = cvae_model.generate(conditions_pyramid, device=DEVICE).detach().cpu().numpy()
                np.save(f"{out_path}/flow_generated_e{num_epoch}.npy", flow_hat_generated)
                np.save(f"{out_path}/condition.npy", conditions_pyramid[-1].detach().cpu().numpy())

    torch.save(cvae_model.state_dict(), f"{out_path}/model_final.pt")
    cvae_model.eval()
    for batch_num, (flow, conditions_pyramid) in enumerate(val_loader):        
        flow_hat_generated = cvae_model.generate(conditions_pyramid, device=DEVICE).detach().cpu().numpy()
        np.save(f"{out_path}/flow_generated_final.npy", flow_hat_generated)
        np.save(f"{out_path}/condition.npy", conditions_pyramid[-1].detach().cpu().numpy())


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # torch.manual_seed(21)
    DEVICE0 = "cuda:0"#"cuda")
    DEVICE1 = "cuda:1"#"cuda")

    top_out_path = f"/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(top_out_path)
    
    config_file_path = "/home/shahar/projects/CVAE_proj/CVAE/config.json"
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    output_config_path = os.path.join(top_out_path, "config.json")
    with open(output_config_path, 'w') as output_config_file:
        json.dump(config, output_config_file, indent=4)
    # l1o_idx, device = 2, DEVICE0
    # train_cvae(l1o_idx, top_out_path, device, config)
    
    for batch in range(11):
        processes = []
        for l1o_idx, device in enumerate([DEVICE0, DEVICE0, DEVICE1, DEVICE1]):
            print("l1o_idx", (batch*4)+l1o_idx)
            p = mp.Process(target=train_cvae, args=((batch*4)+l1o_idx, top_out_path, device, config))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print(1)

