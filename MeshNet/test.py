import numpy as np
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import ModelNet40
from models import MeshNet
from utils import append_feature, calculate_map
root_path = '/content/drive/MyDrive/DL_diamond_cutting/MeshNet/'

cfg = get_test_config(root_path)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
use_gpu = torch.cuda.is_available()

data_set = ModelNet40(cfg=cfg['dataset'], root_path=root_path, part='test')
data_loader = data.DataLoader(data_set, batch_size=1, num_workers=4, shuffle=False, pin_memory=False)


def test_model(model):

    criterion = nn.MSELoss()
    running_loss = 0.0

    for i, (centers, corners, normals, neighbor_index, targets, impurity_label) in enumerate(data_loader):
        if use_gpu:
            centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
            corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
            normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
            neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))
            targets = Variable(torch.cuda.FloatTensor(targets.cuda()))
            impurity_label = Variable(torch.cuda.FloatTensor(impurity_label.cuda()))
        else:
            centers = Variable(torch.FloatTensor(centers))
            corners = Variable(torch.FloatTensor(corners))
            normals = Variable(torch.FloatTensor(normals))
            neighbor_index = Variable(torch.LongTensor(neighbor_index))
            targets = Variable(torch.FloatTensor(targets))
            impurity_label = Variable(torch.FloatTensor(impurity_label))

        outputs, feas = model(centers, corners, normals, neighbor_index, impurity_label)
        loss = criterion(outputs, targets)
        test_file_path, _ = data_set.data[i]
        test_file_label = test_file_path.split('.')[0] + "_prediction.npy"
        np.save(test_file_label, outputs.detach().cpu().clone().numpy())
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(data_set)

    print('Loss: {:.4f}'.format(float(epoch_loss)))


if __name__ == '__main__':

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    if use_gpu:
        model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(root_path,cfg['load_model'])))
    model.eval()

    test_model(model)
