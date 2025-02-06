import os
import sys
import math
sys.path.insert(0, os.getcwd())
import torch
from torch import nn
from einops.layers.torch import Rearrange
from utils.miscellaneous import get_dct_matrix


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):   
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()

        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPblock(nn.Module):
    def __init__(self, dim, seq):
        super().__init__()

        self.fc = Temporal_FC(seq)

        # self.norm = nn.BatchNorm1d(dim)
        self.norm = LN(dim)
        # self.norm = nn.LayerNorm(seq)

        # self.activation_fn = nn.Tanh()
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc.fc.bias, 0)

    def forward(self, x):

        x_ = self.fc(x)
        # x_ = self.activation_fn(x_)
        x_ = self.norm(x_)
        x = x + x_

        return x

class TransMLP(nn.Module):
    def __init__(self, dim, seq, num_layers):
        super().__init__()
        self.mlps = nn.Sequential(*[MLPblock(dim, seq) for _ in range(num_layers)])

    def forward(self, x):
        x = self.mlps(x)
        return x

def build_mlps(motion_dim, seq_length, layers=48):

    return TransMLP(
        dim= motion_dim,
        seq= seq_length,
        num_layers= layers,
    )

class siMLPe(nn.Module):
    def __init__(self, model_param):
        super(siMLPe, self).__init__()
        self.arr0 = Rearrange('b seq_length motion_dim -> b motion_dim seq_length')
        self.arr1 = Rearrange('b motion_dim seq_length -> b seq_length motion_dim')

        dct_m, idct_m = get_dct_matrix(model_param['N'])
        dct_m, idct_m = torch.from_numpy(dct_m).float().unsqueeze(0), torch.from_numpy(idct_m).float().unsqueeze(0)
        self.register_buffer('dct_m', dct_m)
        self.register_buffer('idct_m', idct_m)

        self.motion_fc_in = nn.Linear(model_param['motion_dim'], model_param['motion_dim'])
        self.motion_mlp = build_mlps(model_param['motion_dim'], model_param['N'])
        self.motion_fc_out = nn.Linear(model_param['motion_dim'], model_param['motion_dim'])
        # self.last_layer = torch.nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_in.weight, gain=1e-8)
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_in.bias, 0)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):

        motion_ = torch.matmul(self.dct_m, motion_input)
        motion_ = self.motion_fc_in(motion_)
        motion_ = self.arr0(motion_)
        motion_ = self.motion_mlp(motion_)
        motion_ = self.arr1(motion_)
        motion_ = self.motion_fc_out(motion_)
        motion_ = torch.matmul(self.idct_m, motion_) 
        # motion_ = self.last_layer(motion_)
        return motion_
    
class siMLPe_(nn.Module):
    def __init__(self, model_param):
        super(siMLPe_, self).__init__()
        self.arr0 = Rearrange('b seq_length motion_dim -> b motion_dim seq_length')
        self.arr1 = Rearrange('b motion_dim seq_length -> b seq_length motion_dim')

        dct_m, idct_m = get_dct_matrix(model_param['N'])
        dct_m, idct_m = torch.from_numpy(dct_m).float().unsqueeze(0), torch.from_numpy(idct_m).float().unsqueeze(0)
        self.register_buffer('dct_m', dct_m)
        self.register_buffer('idct_m', idct_m)

        self.motion_fc_in = nn.Linear(model_param['motion_dim'], model_param['motion_dim'])
        self.motion_mlp = build_mlps(model_param['motion_dim'], model_param['N'], 24)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_in.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_in.bias, 0)

    def forward(self, motion_input):

        motion_ = torch.matmul(self.dct_m, motion_input)
        motion_ = self.motion_fc_in(motion_)
        motion_ = self.arr0(motion_)
        motion_ = self.motion_mlp(motion_)
        motion_ = self.arr1(motion_)
        return motion_

class Popeyes(nn.Module):
    def __init__(self, f_size, height, width, latent_size):
        super().__init__()
        self.lower_dimensional_repre = False
        self.h = height
        self.w = width
        self.f_size = f_size
        self.pool_1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU(inplace=True))
        
        self.conv1_branch_2 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(128, 128, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(128, 256, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv1_branch_1 = nn.Sequential(nn.Conv3d(64, 256, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv2 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(128, 128, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(128, 256, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(256))
        
        self.conv3_branch_2 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(128, 128, kernel_size=(3), stride=(1), padding=1),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(128, 512, kernel_size=(1), stride=(1), padding=0),
                                            nn.BatchNorm3d(512))
        
        self.conv3_branch_1 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=0),
                                            nn.BatchNorm3d(512))
        
        self.conv4_branch_2 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=(3, 1, 1), stride=2, padding=(1,0,0)),
                                            nn.BatchNorm3d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm3d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(256, latent_size, kernel_size=(2,1,1), stride=1, padding=0),
                                            nn.BatchNorm3d(latent_size))
        
        self.conv4_branch_1 = nn.Sequential(nn.Conv3d(512, latent_size, kernel_size=(3, 1, 1), stride=2, padding=0),
                                            nn.BatchNorm3d(latent_size))
        self.relu = nn.ReLU(inplace=True)
        self.DynSize = self.dynamic_size(torch.zeros(1, 3, self.f_size, self.h, self.w))
        self.pool_2 = nn.AvgPool3d(tuple(self.DynSize), stride=(1))
        if self.h == 128:
            self.convt3d_1 = nn.ConvTranspose3d(latent_size, 512, kernel_size=3, stride=2, padding=1, output_padding=(0,1,1))
            self.convt3d_2 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.convt3d_3 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=(0,1,1), output_padding=1)
            self.convt3d_4 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=(0,1,1), output_padding=1)
            self.convt3d_5 = nn.ConvTranspose3d(64, 64, kernel_size=(4, 3, 3), stride=(1,2,2), padding=1, output_padding=(0,1,1))
            self.convt3d_6 = nn.ConvTranspose3d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.convt3d_7 = nn.ConvTranspose3d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif self.h == 96:
            self.convt3d_1 = nn.ConvTranspose3d(latent_size, 512, kernel_size=3, stride=2, padding=1, output_padding=(0,1,1))
            self.convt3d_2 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,0,1))
            self.convt3d_3 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=(0,1,1), output_padding=1)
            self.convt3d_4 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.convt3d_5 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=(1,2,2), padding=1, output_padding=(0,1,1))
            self.convt3d_6 = nn.ConvTranspose3d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.convt3d_7 = nn.ConvTranspose3d(3, 3, kernel_size=3, stride=(1,2,2), padding=1, output_padding=(0,1,1))

        self.apply(self.init_weights)

    def forward(self, input):
        # inputs shpae: (4, 3, 24, 96, 128) ->(bs, C, f_size, H, W)
        input = self.pool_1(input) # shpae: (4, 64, 24, 47, 63) ->(bs, C, f_size, H, W)
        out = self.conv1(input) # shpae: (4, 64, 23, 24, 32) ->(bs, C, f_size, H, W)
        out1_b_2 = self.conv1_branch_2(out) # shpae: (4, 128, 23, 24, 32) ->(bs, C, f_size, H, W)

        out1_b_2 += self.conv1_branch_1(out)
        out1_b_2 = self.relu(out1_b_2) # shpae: (4, 256, 23, 24, 32) ->(bs, C, f_size, H, W)

        out2_b_2 = self.conv2(out1_b_2) # shpae: (4, 256, 23, 24, 32) ->(bs, C, f_size, H, W)

        out2_b_2 += out1_b_2
        out2_b_2 = self.relu(out2_b_2) # shpae: (4, 256, 23, 24, 32) ->(bs, C, f_size, H, W)

        out3_b_2 = self.conv3_branch_2(out2_b_2) # shpae: (4, 512, 21, 12, 16) ->(bs, C, f_size, H, W)

        out3_b_2 += self.conv3_branch_1(out2_b_2)
        out3_b_2 = self.relu(out3_b_2) # shpae: (4, 512, 21, 12, 16) ->(bs, C, f_size, H, W)
        
        out4_b_2 = self.conv4_branch_2(out3_b_2) # shpae: (4, 1024, 10, 6, 8) ->(bs, C, f_size, H, W)

        out4_b_2 += self.conv4_branch_1(out3_b_2)
        out4_b_2 = self.relu(out4_b_2) # shpae: (4, 1024, 10, 6, 8) ->(bs, C, f_size, H, W)

        ext_features = self.pool_2(out4_b_2) # shpae: (4, 1024, 1, 1, 1) ->(bs, C, f_size, H, W)
        if self.lower_dimensional_repre:
            return ext_features
        else:
            reconst_img = self.convt3d_1(ext_features) # tensor(batch, 512, 1, 2, 2)
            reconst_img = self.convt3d_2(reconst_img) # tensor(batch, 256, 2, 4/3, 4)
            reconst_img = self.convt3d_3(reconst_img) # tensor(batch, 256, 6, 8/6, 8)
            reconst_img = self.convt3d_4(reconst_img) # tensor(batch, 64, 14/12, 16/12, 16)
            reconst_img = self.convt3d_5(reconst_img) # tensor(batch, 64, 15/12, 32/24, 32)
            reconst_img = self.convt3d_6(reconst_img) # tensor(batch, 3, 30/24, 64/48, 64)
            reconst_img = self.convt3d_7(reconst_img) # tensor(batch, 3, 60/24, 128/96, 128)
            return reconst_img
    
    def dynamic_size(self, input):
        input = self.pool_1(input)
        out = self.conv1(input)
        out1_b_2 = self.conv1_branch_2(out) 

        out1_b_2 += self.conv1_branch_1(out)
        out1_b_2 = self.relu(out1_b_2) # (1, 256, 9, 24, 32)

        out2_b_2 = self.conv2(out1_b_2)

        out2_b_2 += out1_b_2
        out2_b_2 = self.relu(out2_b_2) # (1, 256, 9, 24, 32)

        out3_b_2 = self.conv3_branch_2(out2_b_2)

        out3_b_2 += self.conv3_branch_1(out2_b_2)
        out3_b_2 = self.relu(out3_b_2) # (1, 512, 7, 12, 16)
        
        out4_b_2 = self.conv4_branch_2(out3_b_2)

        out4_b_2 += self.conv4_branch_1(out3_b_2)
        out4_b_2 = self.relu(out4_b_2) # (1, 1024, 6, 6, 8)

        return out4_b_2.shape[2:]
    
    def latent_repre_size(self):
        return self.DynSize
    
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=0)
        elif isinstance(layer, nn.Conv3d):
            nn.init.kaiming_uniform_(layer.weight, a=0)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer.bias, -bound, bound)
        elif isinstance(layer, nn.ConvTranspose3d):
            nn.init.kaiming_uniform_(layer.weight, a=0)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer.bias, -bound, bound)

    def return_features(self):
        self.lower_dimensional_repre = True

class Traj(nn.Module):
    def __init__(self, ps, model_param, featuer_extraction=False):
        super(Traj, self).__init__()
        in_features = ps*model_param['motion_dim_Traj']
        out_features = int(in_features/2)
        if model_param['dataset_name'] == 'vir_poppy':
            self.bottleneck = 300
        else:
            self.bottleneck = 100

        features = []
        while (out_features > self.bottleneck):
            features.append((in_features, out_features))
            in_features = int(in_features/2)
            out_features = int(out_features/2)
        features.append((in_features, self.bottleneck))
        if not featuer_extraction:
            features.extend([tuple(reversed(i)) for i in list(reversed(features))])

        mlp = []
        for tp in features:
            mlp.append(nn.Linear(tp[0], tp[1]))
            mlp.append(nn.Tanh())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, traj):
        ipt = traj.flatten(start_dim=1)
        return self.mlp(ipt)

class quickMLP(nn.Module):
    def __init__(self, ps, model_param):
        super().__init__()
        in_features = model_param['N']*model_param['img_height']*model_param['img_width']*3 + model_param['N']*model_param['motion_dim'] + ps*model_param['motion_dim_Traj']
        mlp = []
        mlp.append(nn.Linear(2964240, 1000))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(1000, 1))
        mlp.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, traj):
        return self.mlp(traj)

if __name__ == '__main__':
    # b, seq, motion_dm = 5, 60, 42
    # x = torch.ones((b,motion_dm,seq))
    # layer = nn.BatchNorm1d(seq)
    # y = layer(x)
    # print(y)

    # arr0 = Rearrange('b seq_length motion_dim -> b motion_dim seq_length')
    # arr1 = Rearrange('b motion_dim seq_length -> b seq_length motion_dim')

    # ipt_tensor = torch.stack([torch.ones(3,4)*(i+1) for i in range(2)])
    # print('Original shape:', ipt_tensor.shape)
    # print('Switch last two dim:', arr0(ipt_tensor).shape)
    # print('Switch back to original shape using arr1:', arr1(arr0(ipt_tensor)).shape)
    # print('Switch back to original shape using arr0:', arr0(arr0(ipt_tensor)).shape)
    # print('Switch dim using arr1:', arr1(arr1(ipt_tensor)).shape) # arr0 is enough for switch dimensions
    # print('Is my input same as before?', (arr1(arr0(ipt_tensor))==ipt_tensor).all())

    # from hp import hyperparameters_virtual
    # toy = siMLPe(hyperparameters_virtual)
    # print(toy.state_dict().keys())
    # print(toy.arr0)
    # print(toy.arr1)

    # x = torch.ones(b,3,seq,128,128).float()
    # toy = Popeyes(seq,128,128,256)
    # out = toy(x)
    # print(out.shape)
    
    # from hp import hyperparameters_real
    # x = torch.ones(b,100,42)
    # toy = Traj(100, {'motion_dim_Traj': 42, 'dataset_name': 'real_poppy'})
    # print(toy(x))

    from hp import hyperparameters_real
    x = torch.rand(64, 2964240)
    toy = quickMLP(10, hyperparameters_real)
    print(toy(x))