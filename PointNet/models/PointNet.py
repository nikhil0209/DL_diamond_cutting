import torch
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm, Tanh, Dropout
import torch.nn.functional as F
import math
# the "MLP" block that you will use the in the PointNet and CorrNet modules you will implement
# This block is made of a linear transformation (FC layer), 
# followed by a Leaky RelU, a Group Normalization (optional, depending on enable_group_norm)
# the Group Normalization (see Wu and He, "Group Normalization", ECCV 2018) creates groups of 32 channels
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)    
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(0.2), GroupNorm(num_groups[i], channels[i]))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(0.2))
                     for i in range(1, len(channels))])


# PointNet module for extracting point descriptors
# num_input_features: number of input raw per-point or per-vertex features 
# 		 			  (should be 3, since we have 3D point positions in this assignment)
# num_output_features: number of output per-point descriptors (should be 32 for this assignment)
# this module should include
# - a MLP that processes each point i into a 128-dimensional vector f_i
# - another MLP that further processes these 128-dimensional vectors into h_i (same number of dimensions)
# - a max-pooling layer that collapses all point features h_i into a global shape representaton g
# - a concat operation that concatenates (f_i, g) to create a new per-point descriptor that stores local+global information
# - a MLP that transforms this concatenated descriptor into the output 32-dimensional descriptor x_i
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class PointNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(PointNet, self).__init__()
        self.mlp1 = MLP([num_input_features, 32, 64, 128, 256, 512])
        self.linear = Lin(512, num_output_features)
    def forward(self, x):
        mlp1_output = self.mlp1(x)
        linear_output = self.linear(mlp1_output)
        global_feature,_ = torch.max(linear_output, 0)
        translations = torch.unsqueeze(F.tanh(global_feature[0:3]), 0)
        rotations = torch.unsqueeze((2*math.pi)*F.sigmoid(global_feature[3:6]), 0)
        scale = torch.reshape(torch.unsqueeze(F.sigmoid(global_feature[-1]), 0), (1, 1)) 
        out = torch.cat((translations, rotations, scale), 1)
        return out

