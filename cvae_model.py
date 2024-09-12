import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True,
         conv_module=nn.Conv3d):  # nn.ConvTranspose3d
    if isReLU:
        return nn.Sequential(
            conv_module(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, dilation=dilation,
                        padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class FeatureExtractor(nn.Module):  # the frozen extractor
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class Encoder(nn.Module):
    def __init__(self, num_chs, max_var=6):
        super(Encoder, self).__init__()
        self.num_chs = num_chs
        self.max_var = max_var
        self.epsilon = 1e-8
        self.convs = nn.ModuleList()
        kernel_size = 3
        stride = 2
        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            if l == 0:
                ch_in_expanded = ch_in + 1
            else:
                ch_in_expanded = ch_in * 2
            if l == 5:
                kernel_size = 1
                stride = 1
            layer = nn.Sequential(
                conv(ch_in_expanded, ch_out, stride=stride, kernel_size=kernel_size),
                conv(ch_out, ch_out, kernel_size=kernel_size)
            )
            self.convs.append(layer)

        self.mean = nn.Conv3d(num_chs[-1], num_chs[-1], kernel_size=1, stride=1, dilation=1, padding=0 // 2, bias=True)
        self.var = nn.Conv3d(num_chs[-1], num_chs[-1], kernel_size=1, stride=1, dilation=1, padding=0 // 2, bias=True)

        self.kl = 0

    def forward(self, flow, condition_pyramid):
        x = flow
        for conv, condition_features in zip(self.convs, condition_pyramid[::-1]):  # check
            x = torch.concat((x, condition_features), dim=1)
            x = conv(x)

        mean = self.mean(x)
        std = torch.exp(0.5 * torch.clamp(self.var(x), max=self.max_var))
        eps = torch.randn_like(std)

        z = mean + eps * std

        self.kl = (std ** 2 + mean ** 2 - torch.log(std + self.epsilon) - 1 / 2).mean()

        return z


class Decoder(nn.Module):
    def __init__(self, num_chs, max_flow_hat_abs_val):
        super(Decoder, self).__init__()
        self.num_chs = num_chs
        self.max_f_hat_abs_val = abs(max_flow_hat_abs_val)
        self.convs = nn.ModuleList()
        kernel_size = 3
        stride = 2
        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            if l == 5:
                ch_in_expanded = ch_in + 1
                kernel_size = 1
                stride = 1
            else:
                ch_in_expanded = ch_in * 2
            layer = nn.Sequential(
                conv(ch_in_expanded, ch_out, stride=stride, kernel_size=kernel_size, conv_module=nn.ConvTranspose3d),
                conv(ch_out, ch_out, kernel_size=kernel_size, conv_module=nn.ConvTranspose3d, isReLU=False)
            )
            if l == 0:
                layer[0][0].output_padding = (1, 1, 0)
            if l == 1:
                layer[0][0].output_padding = (0, 0, 1)
            if l == 2:
                layer[0][0].output_padding = (1, 1, 0)
            if l == 3:
                layer[0][0].output_padding = (0,0,1)
            if l == 4:
                layer[0][0].output_padding = (1,1,1)#(0, 0, 1)
            self.convs.append(layer)

    def forward(self, z, condition_pyramid):
        x = z
        for conv, condition_features in zip(self.convs, condition_pyramid):  # check
            x = torch.concat((x, condition_features), dim=1)
            x = conv(x)
        x = torch.clip(x, min=-self.max_f_hat_abs_val, max=self.max_f_hat_abs_val)
        return x


class CVAE(torch.nn.Module):
    def __init__(self, num_chs, max_flow_hat_abs_val=50):
        super().__init__()
        self.encoder = Encoder(num_chs=num_chs)
        self.decoder = Decoder(num_chs=num_chs[:-1][::-1] + [3], max_flow_hat_abs_val=max_flow_hat_abs_val)

    def forward(self, flow, conditions_pyramid):
        z = self.encoder(flow.float(), conditions_pyramid)
        flow_hat = self.decoder(z, conditions_pyramid)
        return flow_hat, self.encoder.kl

    def generate(self, conditions_pyramid, device, latent_shape=(1,128,3,3,3)):
        z = torch.randn(latent_shape).to(device)
        flow_hat = self.decoder(z, conditions_pyramid)
        return flow_hat

    # def save_model(self, path, epoch):
    #     try:
    #         models = {'epoch': epoch, 'state_dict_encoder': self.encoder.module.state_dict(), 'state_dict_decoder': self.decoder.module.state_dict()}
    #     except:
    #         models = {'epoch': epoch, 'state_dict_encoder': self.encoder.state_dict()}
    #     save_checkpoint(os.path.join(self.output_root , "checkpoints"), models, name, is_best) 
    #     torch.save(states, file_path)

