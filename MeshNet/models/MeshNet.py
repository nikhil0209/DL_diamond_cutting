import torch
import torch.nn as nn
from models import SpatialDescriptor, StructuralDescriptor, MeshConvolution


class MeshNet(nn.Module):

    def __init__(self, cfg, require_fea=False):
        super(MeshNet, self).__init__()
        self.require_fea = require_fea

        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(cfg['structural_descriptor'])
        self.mesh_conv1 = MeshConvolution(cfg['mesh_convolution'], 64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(cfg['mesh_convolution'], 256, 256, 512, 512)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 9)
        )

    def forward(self, centers, corners, normals, neighbor_index, impurity_label):
        centers = torch.cat((centers,impurity_label),dim=1)
        corners = torch.cat((corners,impurity_label),dim=1)
        #normals = torch.cat((normals,impurity_label),dim=1)
        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index, impurity_label)

        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, neighbor_index)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, neighbor_index)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        fea = self.concat_mlp(torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))
        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)
        fea = self.classifier[:-1](fea)
        cls_ = self.classifier[-1:](fea)

        if self.require_fea:
            return cls_, fea / torch.norm(fea)
        else:
            return cls_
