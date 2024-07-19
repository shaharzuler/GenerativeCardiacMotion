import os
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from four_d_ct_cost_unrolling import torch_to_np
from flow_n_corr_utils import min_max_norm

from cvae_model import CVAE
from dataset import cvae_dataset


DEVICE = torch.device("cuda")

out_path = f"/home/fiman/projects/DL_course/spectral_diffusion/spectral_diffusion/outputs/{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(out_path)
writer = SummaryWriter(out_path)

train_set = cvae_dataset("/media/fiman/storage/datasets/DL_course/cvae_cardio_flow_dataset", device=DEVICE)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

num_chs = [3, 16, 32, 64, 96, 128, 128]
cvae_model = CVAE(num_chs=num_chs).to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(cvae_model.parameters(), lr=1e-3)

lambda_reconstraction = 0.5
lambda_kl = 0.5

num_epochs = 100
for num_epoch in range(num_epochs):
    cvae_model.train()
    epoch_acc_loss = 0
    epoch_reconstraction_loss = 0
    epoch_kl = 0
    for batch_num, (flow, conditions_pyramid) in enumerate(train_loader):
        optimizer.zero_grad()
        flow_hat, kl = cvae_model(flow, conditions_pyramid)
        reconstraction_loss = criterion(flow_hat, flow)
        loss = lambda_reconstraction * reconstraction_loss + lambda_kl * kl
        loss.backward()
        optimizer.step()
        epoch_acc_loss += loss.item()
        epoch_kl += kl.item()
        epoch_reconstraction_loss += reconstraction_loss.item()
    epoch_acc_loss /= (batch_num+1)
    epoch_reconstraction_loss /= (batch_num+1)
    epoch_kl /= (batch_num+1)
    print(f"reconstraction_loss {epoch_reconstraction_loss} kl {epoch_kl} total {epoch_acc_loss}")
    writer.add_scalar("reconstraction_loss", epoch_reconstraction_loss, num_epoch)
    writer.add_scalar("kl", epoch_kl, num_epoch)
    writer.add_scalar("total loss", epoch_acc_loss, num_epoch)
    writer.add_image("f_hat",min_max_norm(flow_hat[0, :, :, 60, :]), num_epoch) # 3 h w
