import numpy as np
import os
import torch
import torch.utils.data as data

type_to_index_map = {'diamond':1}

class UNetData(data.Dataset):

    def __init__(self, cfg, root_path, part='train'):
        self.root = os.path.join(root_path, cfg['data_root'])
        self.grid_size = cfg['grid']
        self.part = part

        self.data = []
        for type in os.listdir(self.root):
            if type=='.DS_Store':
                continue
            type_index = type_to_index_map[type]
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append((os.path.join(type_root, filename), type_index))

    def __getitem__(self, i):
        path, type = self.data[i]
        data = np.load(path)
        outer = data['outer']
        rocks = data['rocks']
        targets = data['targets']
        pitch = data['pitch']
        radius = data['radius']
        diamond_center_grid_point = data['diamoind_center_grid_point']


        #combine outer and rocks
        input = outer*1 + rocks*2
        resolution = outer.shape[0]+1
        #make sure the grids are correct size
        target = np.zeros((1,resolution,resolution,resolution))
        if np.random.uniform(0,1)>0.5:
            target[:,1:,1:,1:] = input
        else:
            target[:,:-1,:-1,:-1] = input
        input = target
        # to tensor
        input = torch.from_numpy(input).float()
        diamond_center_grid_point = torch.from_numpy(diamond_center_grid_point).long()
        targets = torch.from_numpy(targets).float()
        
        return input, diamond_center_grid_point, targets, pitch, radius

    def __len__(self):
        return len(self.data)
