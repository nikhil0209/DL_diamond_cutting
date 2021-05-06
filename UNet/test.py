import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import UNetData
from models import UNet3D
from utils import get_unit_diamond_vertices, save_loss_plot, regression_classification_loss, point_wise_L1_loss, axis_aligned_miou, get_output_from_prediction

root_path = '../UNet/'

cfg = get_test_config(root_path)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
use_gpu = torch.cuda.is_available()

data_set = UNetData(cfg=cfg['dataset'], root_path=root_path, part='test')
data_loader = data.DataLoader(data_set, batch_size=1, num_workers=4, shuffle=False, pin_memory=False)

def test_model(model):
    model.eval()    
    criterion = nn.L1Loss()
    running_loss = 0.0
    running_l1_loss = 0.0
    running_scale_loss = 0.0
    running_center_loss = 0.0
    running_rotation_loss = 0.0
    running_miou = 0.0
    unit_diamond_vertices = get_unit_diamond_vertices(root_path)
    if use_gpu:
        unit_diamond_vertices = Variable(torch.cuda.FloatTensor(unit_diamond_vertices.cuda()))
    else:
        unit_diamond_vertices = Variable(torch.FloatTensor(unit_diamond_vertices))

    for i, (input, diamond_center_grid_point, targets, pitch, radius) in enumerate(data_loader):
        if use_gpu:
            input = Variable(torch.cuda.FloatTensor(input.cuda()))
            diamond_center_grid_point = Variable(torch.cuda.LongTensor(diamond_center_grid_point.cuda()))
            targets = Variable(torch.cuda.FloatTensor(targets.cuda()))
            pitch = torch.tensor(pitch).to(targets)
            radius = torch.tensor(radius).to(targets)
        else:
            input = Variable(torch.FloatTensor(input))
            diamond_center_grid_point = Variable(torch.LongTensor(diamond_center_grid_point))
            targets = Variable(torch.FloatTensor(targets))
            pitch = torch.tensor(pitch).to(targets)
            radius = torch.tensor(radius).to(targets)
        with torch.no_grad():
            center_probs, pred_rot_scale = model(input,return_encoder_features = True)
        model_loss = regression_classification_loss(center_probs, pred_rot_scale, diamond_center_grid_point, targets[:,3:], alpha=0.5)
        outputs = get_output_from_prediction(center_probs, pred_rot_scale, pitch, radius)
        loss = point_wise_L1_loss(outputs, targets, unit_diamond_vertices)
        l1_loss = criterion(outputs, targets)
        scale_loss = criterion(outputs[:,-1:],targets[:,-1:])
        center_loss = criterion(outputs[:,:3],targets[:,:3])
        rotation_loss = criterion(outputs[:,3:6],targets[:,3:6])
        miou = axis_aligned_miou(outputs,targets)

        test_file_path, _ = data_set.data[i]
        test_file_label = test_file_path.split('.')[0] + "_prediction.npy"
        np.save(test_file_label, outputs.detach().cpu().clone().numpy())
        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_scale_loss += scale_loss.item()
        running_center_loss += center_loss.item()
        running_rotation_loss += rotation_loss.item()
        running_miou += miou.item()
        
    epoch_loss = running_loss / len(data_set)
    epoch_l1_loss = running_l1_loss / len(data_set)
    epoch_scale_loss = running_scale_loss / len(data_set)
    epoch_center_loss = running_center_loss / len(data_set)
    epoch_rotation_loss = running_rotation_loss / len(data_set)
    epoch_miou = running_miou / len(data_set)

    print('Loss: {:.4f}'.format(float(epoch_loss)))
    print('L1 Loss: {:.4f}'.format(float(epoch_l1_loss)))
    print('Scale L1 Loss: {:.4f}'.format(float(epoch_scale_loss)))
    print('Center L1 Loss: {:.4f}'.format(float(epoch_center_loss)))
    print('M IOU: {:.4f}'.format(float(epoch_miou)))


if __name__ == '__main__':

    model_cfg = cfg['UNet'] 
    model = UNet3D(model_cfg['in_channels'], model_cfg['out_channels'], final_sigmoid=model_cfg['final_sigmoid'], f_maps=model_cfg['f_maps'], layer_order=model_cfg['layer_order'], num_groups=model_cfg['num_groups'], is_segmentation=model_cfg['is_segmentation'])
    if use_gpu:
        model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(root_path,cfg['load_model'])))
    model.eval()

    test_model(model)
