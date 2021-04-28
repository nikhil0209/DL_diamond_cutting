import numpy as np
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import ModelDataset
from models import PointNet
#from train import stochastic_loss
from train import point_wise_L1_loss
from train import get_unit_diamond_vertices

root_path = '/content/drive/MyDrive/DiamondPrediction/PointNet/'

cfg = get_test_config(root_path)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
use_gpu = torch.cuda.is_available()

data_set = ModelDataset(cfg=cfg['dataset'], root_path=root_path, part='test')
data_loader = data.DataLoader(data_set, batch_size=1, num_workers=4, shuffle=False, pin_memory=False)

def test_model(model):
    
    criterion = nn.L1Loss()
    running_loss = 0.0
    unit_diamond_vertices = get_unit_diamond_vertices()
    for i, (rock_vertices, targets) in enumerate(data_loader):
        if use_gpu:
            rock_vertices = Variable(torch.cuda.FloatTensor(rock_vertices.cuda()))
            targets = Variable(torch.cuda.FloatTensor(targets.cuda()))
            unit_diamond_vertices = Variable(torch.cuda.FloatTensor(unit_diamond_vertices.cuda()))
        else:
            rock_vertices = Variable(torch.FloatTensor(rock_vertices))
            targets = Variable(torch.FloatTensor(targets))
            unit_diamond_vertices = Variable(torch.FloatTensor(unit_diamond_vertices))

        rock_vertices = torch.reshape(rock_vertices, (-1, 3))
        outputs = model(rock_vertices)
        loss = point_wise_L1_loss(outputs, targets, unit_diamond_vertices)
        test_file_path = data_set.data[i]
        test_file_label = test_file_path.split('.')[0] + "_prediction.npy"
        np.save(test_file_label, outputs.detach().cpu().clone().numpy())
        running_loss += loss.item()*targets.size(0)
    
    epoch_loss = running_loss / len(data_set)

    print('Loss: {:.4f}'.format(float(epoch_loss)))


if __name__ == '__main__':

    model = PointNet(3, 7)
    if use_gpu:
        model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(root_path,cfg['load_model'])))
    model.eval()
    test_model(model)
