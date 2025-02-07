import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.autograd import Variable

class ExponentialUnit(nn.Module):
    def __init__(self):
        super(ExponentialUnit, self).__init__()
    
    def forward(self, x):
        return torch.exp(x)
    
"""=====================================
        Base-CNN
====================================="""
class CNN(nn.Module):
    def __init__(self, spatial_shape, mask, device='cpu'):
        super(CNN, self).__init__()
        self.downscale_factor = 1/8
        
        """ Land mask: original 2D size"""
        self.mask = torch.Tensor(mask)[None, None, :, :].to(device)
        """ Land mask: downscaled 2D size"""
        self.mask_dscale = F.interpolate(self.mask, mode='nearest',
                                         scale_factor = self.downscale_factor)
        self.mask_dscale = (self.mask_dscale > 0.5).int()
        
        """ Convolution reduce X dimension by (8 x 8)"""
        self.spatial_extractor = nn.Sequential(
                                      nn.Conv2d(1, 8, kernel_size=5, dilation=1, padding='same'), 
                                      nn.Conv2d(8, 8, kernel_size=5, dilation=1, padding='same'), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d((2,2), stride=2),
                                      nn.Conv2d(8, 16, kernel_size=5, dilation=1, padding='same'),
                                      nn.Conv2d(16, 16, kernel_size=5, dilation=1, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,2), stride=2), 
                                      nn.Conv2d(16, 16, kernel_size=5, dilation=1, padding='same'),
                                      nn.Conv2d(16, 16, kernel_size=5, dilation=1, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,2), stride=2), 
                                      nn.Conv2d(16, 8, kernel_size=5, dilation=1, padding='same'),
                                      nn.Conv2d(8, 1, kernel_size=5, dilation=1, padding='same'),
                                      nn.ReLU(),
                                      # ExponentialUnit(),
                                      nn.Dropout(p=0.2))
        
        """ MLP layer"""
        self.num_features = int(spatial_shape[0] * self.downscale_factor) * \
                            int(spatial_shape[1] * self.downscale_factor)
        self.mlp = nn.Sequential(nn.Linear(self.num_features, 32), nn.ReLU(), 
                                 nn.Linear(32,1))
        
    def forward(self, x):
        """# 1. Convolutional layers"""
        h = self.spatial_extractor(x)
        h = self.mask_dscale * h
        """# 2. Unroll 2D -> 1D"""
        h = h.view(h.shape[0], -1)
        """# 3. MLP layer"""
        out = self.mlp(h)
        out = torch.sigmoid(out)
        out = out.view(-1)
        
        return out

"""============================================
        NA-CNN: Neural Additive CNN
============================================"""
class NeuralAdditiveCNN(nn.Module):
    def __init__(self, spatial_shape, mask, interpolation=1/2, device='cpu'):
        super(NeuralAdditiveCNN, self).__init__()
        self.interpolation = interpolation
        self.downscale_factor = interpolation * 1/4
        
        """ Land mask: original 2D size"""
        self.mask = torch.Tensor(mask)[None, None, :, :].to(device)
        """ Land mask: downscaled 2D size"""
        self.mask_dscale = F.interpolate(self.mask, mode='nearest',
                                         scale_factor = self.downscale_factor)
        self.mask_dscale = (self.mask_dscale > 0.5).int()
        
        """ Convolution reduce X dimension by (8 x 8)"""
        self.downsample = F.interpolate
        self.spatial_extractor = nn.Sequential(
                                      nn.Conv2d(1, 8, kernel_size=5, dilation=1, padding='same'), 
                                      nn.Conv2d(8, 8, kernel_size=5, dilation=1, padding='same'), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d((2,2), stride=2),
                                      nn.Conv2d(8, 8, kernel_size=5, dilation=1, padding='same'),
                                      nn.Conv2d(8, 8, kernel_size=5, dilation=1, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,2), stride=2), 
                                      # ExponentialUnit(),
                                      nn.Dropout(p=0.7))
        
        """ Feature net"""
        self.num_features = int(spatial_shape[0] * self.downscale_factor) * \
                            int(spatial_shape[1] * self.downscale_factor)
        self.featureNet = nn.ModuleDict({})
        for num in range(self.num_features):
            self.featureNet[f'FeatureNet {num}'] = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), 
                                                                 nn.Dropout(p=0.2),
                                                                 nn.Linear(32, 32), nn.ReLU(), 
                                                                 nn.Linear(32, 1, bias=False))
        
        self.add_bias = nn.Parameter(torch.Tensor([0.]))

    def latent_presigmoid(self, x):
        """# 1. Downsampling"""
        x = self.downsample(x, scale_factor=self.interpolation, 
                            mode='bilinear', align_corners=False)
        
        """# 2. Convolutional layers"""
        h = self.spatial_extractor(x)
        """# 3. Unroll 2D -> 1D"""
        h = h.view(h.shape[0], h.shape[1], -1)
        
        """# 4. Feature Net"""
        feature_stack = []
        for num in range(self.num_features):
            latent = self.featureNet[f'FeatureNet {num}'](h[:,:, num])
            feature_stack.append(latent)
        feature_stack = torch.stack(feature_stack, axis=-1)
        
        """# 5. Neural Additive CNN -> interpretation here"""
        out = feature_stack[:,0,:]
        
        """# 6. Mask out land regions -> suppress weight to 0
           #    sigmoid(0) = 0.5 -> 0 is not discriminative"""
        mask_flatten = self.mask_dscale.view(-1)
        out = out * mask_flatten
        
        return out
        
    def forward(self, x):
        """# 1. Obtain latent representation from convolution & feature net"""
        h = self.latent_presigmoid(x)
        """# 2. Neural Additive Model + bias"""
        out = torch.sum(h, axis=-1) + self.add_bias
        out = torch.sigmoid(out)
        out = out.view(-1)
        
        return out

"""============================================
        NA-FCN: Neural Additive FCN
============================================"""
class NeuralAdditiveFCN(nn.Module):
    def __init__(self, spatial_shape, mask, device='cpu'):
        super(NeuralAdditiveFCN, self).__init__()
        self.downscale_factor = 1/8
        
        """ Land mask: original 2D size"""
        self.mask = torch.Tensor(mask)[None, None, :, :].to(device)
        """ Land mask: downscaled 2D size"""
        self.mask_dscale = F.interpolate(self.mask, mode='nearest',
                                         scale_factor = self.downscale_factor)
        self.mask_dscale = (self.mask_dscale > 0.5).int()
        
        """ Convolution reduce X dimension by (8 x 8)"""
        self.spatial_extractor = nn.Sequential(
                                      nn.Conv2d(1, 8, kernel_size=5, dilation=1, padding='same'), 
                                      nn.Conv2d(8, 8, kernel_size=5, dilation=1, padding='same'), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d((2,2), stride=2),
                                      nn.Conv2d(8, 16, kernel_size=5, dilation=1, padding='same'),
                                      nn.Conv2d(16, 16, kernel_size=5, dilation=1, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,2), stride=2), 
                                      nn.Conv2d(16, 16, kernel_size=5, dilation=1, padding='same'),
                                      nn.Conv2d(16, 16, kernel_size=5, dilation=1, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d((2,2), stride=2), 
                                      nn.Conv2d(16, 8, kernel_size=5, dilation=1, padding='same'),
                                      nn.Conv2d(8, 1, kernel_size=5, dilation=1, padding='same'),
                                      nn.ReLU(),
                                      # ExponentialUnit(),
                                      nn.Dropout(p=0.2))
        
        """ Grid weights"""
        self.num_features = int(spatial_shape[0] * self.downscale_factor) * \
                            int(spatial_shape[1] * self.downscale_factor)
        self.grid_weights = nn.Parameter(torch.zeros(self.num_features))
        
        """ Additive bias"""
        self.add_bias = nn.Parameter(torch.Tensor([0.]))

    def latent_presigmoid(self, x):
        """# 1. Convolutional layers"""
        h = self.spatial_extractor(x)
        """# 2. Unroll 2D -> 1D"""
        h = h.view(h.shape[0], -1)
        
        """# 3. Grid weights -> interpretation here"""
        out = self.grid_weights * h
        
        """# 4. Mask out land regions -> suppress weight to 0
           #    sigmoid(0) = 0.5 -> 0 is not discriminative"""
        mask_flatten = self.mask_dscale.view(-1)
        out = out * mask_flatten
        
        return out
        
    def forward(self, x):
        """# 1. Obtain latent representation from convolution & feature net"""
        h = self.latent_presigmoid(x)
        """# 2. Neural Additive Model + bias"""
        out = torch.sum(h, axis=-1) + self.add_bias
        out = torch.sigmoid(out)
        out = out.view(-1)
        
        return out

"""=========================================
            BASELINE: CNN4
========================================="""
class CNN4(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = F.interpolate
        self.conv1 = nn.Conv2d(1, 8, (5,5), stride=(1,1))
        self.conv2 = nn.Conv2d(8, 16, (5,5), stride=(1,1))
        self.conv3 = nn.Conv2d(16, 32, (5,5), stride=(1,1))
        self.pool1 = nn.MaxPool2d((2,2), stride=2)
        self.conv4 = nn.Conv2d(32, 64, (5,5), stride=(1,1))
        self.pool2 = nn.MaxPool2d((2,2), stride=2)
        self.dropout = nn.Dropout2d(p=0.3)
        self.classifier = nn.Sequential(
                            nn.Linear(6528, 256),
                            nn.Linear(256,1), 
                            nn.Sigmoid())

    def forward(self, x):
        x = self.downsample(x, scale_factor=1/4, mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        x = x.view(-1)

        return x
    
"""=========================================
            BASELINE: LeNet5
========================================="""
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=8880, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        logits = self.classifier(x)
        # logits = logits.view(-1)
        return logits

"""=========================================
            BASELINE: MLP
========================================="""    
class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=n_classes)
        )


    def forward(self, x):
        logits = self.classifier(x)
        logits = torch.sigmoid(logits)
        return logits

    
"""=========================================
        BASELINE: Logsitic Regression
========================================="""
class logisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(5461, 2)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)

        return x
    
"""=========================================
        BASELINE: Spatial Attention
========================================="""
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, hidden_ratio=8):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // hidden_ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_channels // hidden_ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class AttentionNet(nn.Module):
    def __init__(self, mask, device='cpu'):
        super(AttentionNet, self).__init__()
        self.downscale_factor = 1/8
        
        """ Land mask: original 2D size"""
        self.mask = torch.Tensor(mask)[None, None, :, :].to(device)
        """ Land mask: downscaled 2D size"""
        self.mask_dscale = F.interpolate(self.mask, mode='nearest',
                                         scale_factor = self.downscale_factor)
        self.mask_dscale = (self.mask_dscale > 0.5).int()
        
        self.convblock1 = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.ca = ChannelAttention(in_channels=16)
        self.sa = SpatialAttention()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=15840, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        
    def get_spatial_attn(self, x):
        x = self.convblock1(x)
        x = self.ca(x) * x
        out = self.sa(x) * self.mask_dscale
        # spatial attn shape is [N, 1, H, W]
        out = torch.mean(out, axis=1)
        return out


    def forward(self, x):
        x = self.convblock1(x)
        x = self.ca(x) * x
        x = (self.sa(x) * self.mask_dscale) * x
        x = x.view(x.shape[0], -1)
        logits = self.classifier(x)
        # logits = logits.view(-1)
        return logits
    
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if(m.bias is not None):
            m.bias.data.fill_(0.001)