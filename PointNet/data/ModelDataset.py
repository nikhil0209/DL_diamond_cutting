import numpy as np
import os
import torch
import torch.utils.data as data

class ModelDataset(data.Dataset):
    def __init__(self, cfg, root_path, part='train'):
        self.root = os.path.join(root_path, cfg['data_root'])
        self.part = part
        self.data = []
        part_root = os.path.join(self.root, part)
        for filename in os.listdir(part_root):
            if filename.endswith('.npz'):
                self.data.append(os.path.join(part_root, filename))

    def __getitem__(self, i):
        path = self.data[i]
        data = np.load(path)
        vertices = data['vertices']
        targets = data['targets']
        vertices = torch.from_numpy(vertices).float()
        targets = torch.from_numpy(targets).float()
        return vertices, targets

    def __len__(self):
        return len(self.data)
