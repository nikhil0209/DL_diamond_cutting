import copy
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import get_train_config
from data import UNetData
from models import UNet3D
from utils import get_unit_diamond_vertices, save_loss_plot, regression_classification_loss#, point_wise_L1_loss
import numpy as np
from scipy.spatial.transform import Rotation as R

root_path = '../UNet/'

cfg = get_train_config(root_path)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
use_gpu = torch.cuda.is_available()

data_set = {
    x: UNetData(cfg=cfg['dataset'], root_path=root_path, part=x) for x in ['train', 'val']
}
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=False)
    for x in ['train', 'val']
}

def train_model(model, optimizer, scheduler, cfg):

    best_loss = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []
    unit_diamond_vertices = get_unit_diamond_vertices(root_path)
    if use_gpu:
        unit_diamond_vertices = Variable(torch.cuda.FloatTensor(unit_diamond_vertices.cuda()))
    else:
        unit_diamond_vertices = Variable(torch.FloatTensor(unit_diamond_vertices))

    for epoch in range(1, cfg['max_epoch']):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)
        for phrase in ['train', 'val']:

            if phrase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            ft_all, lbl_all = None, None

            for i, (input, diamond_center_grid_point, targets, pitch, radius) in enumerate(data_loader[phrase]):

                optimizer.zero_grad()
                if use_gpu:
                    input = Variable(torch.cuda.FloatTensor(input.cuda()))
                    diamond_center_grid_point = Variable(torch.cuda.LongTensor(diamond_center_grid_point.cuda()))
                    targets = Variable(torch.cuda.FloatTensor(targets.cuda()))
                else:
                    input = Variable(torch.FloatTensor(input))
                    diamond_center_grid_point = Variable(torch.LongTensor(diamond_center_grid_point))
                    targets = Variable(torch.FloatTensor(targets))
                    
                with torch.set_grad_enabled(phrase == 'train'):
                    eps = 1e-12
                    center_probs, pred_rot_scale = model(input,return_encoder_features = True)
                    loss = regression_classification_loss(center_probs, pred_rot_scale, diamond_center_grid_point, targets[:,3:], alpha=0.5)
                    if phrase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * input.size(0)

            epoch_loss = running_loss / len(data_set[phrase])

            if phrase == 'train':
                print('{} Loss: {:.4f}'.format(phrase, epoch_loss))
                train_losses.append(epoch_loss)

            if phrase == 'val':
                val_losses.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch % 1 == 0:
                    torch.save(copy.deepcopy(model.state_dict()), root_path + '/ckpt_root/{}.pkl'.format(epoch))

                print('{} Loss: {:.4f}'.format(phrase, epoch_loss))
        
        save_loss_plot(train_losses,val_losses,root_path)

    return best_model_wts


if __name__ == '__main__':

    model_cfg = cfg['UNet']
    scheduler_cfg = cfg['lr_scheduler']
    model = UNet3D(model_cfg['in_channels'], model_cfg['out_channels'], final_sigmoid=model_cfg['final_sigmoid'], f_maps=model_cfg['f_maps'], layer_order=model_cfg['layer_order'], num_groups=model_cfg['num_groups'], is_segmentation=model_cfg['is_segmentation'])
    if use_gpu:
        model.cuda()
    model = nn.DataParallel(model)
    #model.load_state_dict(torch.load(os.path.join(root_path, cfg['ckpt_root'], 'MeshNet_best.pkl')))
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], betas = tuple((0.9, 0.999)), weight_decay=cfg['weight_decay'])
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_cfg['mode'], factor=scheduler_cfg['factor'],patience=scheduler_cfg['patience'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    for f in os.listdir(root_path + '/ckpt_root/'):
        os.remove(os.path.join(root_path + '/ckpt_root/', f))
    best_model_wts = train_model(model, optimizer, scheduler, cfg)
    torch.save(best_model_wts, os.path.join(root_path, cfg['ckpt_root'], 'UNet_best.pkl'))
