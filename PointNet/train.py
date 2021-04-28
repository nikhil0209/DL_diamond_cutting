import copy
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import get_train_config
from data import ModelDataset
from models import PointNet
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_unit_diamond_vertices():
    vertices = np.transpose(np.genfromtxt('/content/drive/MyDrive/DiamondPrediction/PointNet/data/unit_diamond.txt', delimiter=' '))
    return torch.from_numpy(vertices).float()

root_path = '/content/drive/MyDrive/DiamondPrediction/PointNet/'

cfg = get_train_config(root_path)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
use_gpu = torch.cuda.is_available()

data_set = {
    x: ModelDataset(cfg=cfg['dataset'], root_path=root_path, part=x) for x in ['train', 'val']
}
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=False)
    for x in ['train', 'val']
}

def inverse_distance_loss(outputs, rock_vertices):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inverse_distances = 1/torch.sqrt(torch.sum(torch.square(torch.sub(rock_vertices, outputs[0, 0:3])), axis=1))
    return torch.sum(inverse_distances)

def point_wise_L1_loss(outputs, targets, unit_diamond_vertices):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    r_target = torch.from_numpy(R.from_euler('xyz', [[targets[0, 3], targets[0, 4], targets[0, 5]]]).as_matrix()).float().to(device)
    target_vertices = torch.mul(targets[0, -1], (torch.matmul(r_target, unit_diamond_vertices))) + torch.reshape(targets[0, 0:3], (3, 1))
    r_output = torch.from_numpy(R.from_euler('xyz', [[outputs[0, 3], outputs[0, 4], outputs[0, 5]]]).as_matrix()).float().to(device)
    output_vertices = torch.mul(outputs[0, -1], (torch.matmul(r_output, unit_diamond_vertices))) + torch.reshape(outputs[0, 0:3], (3, 1))
    return nn.L1Loss(reduction='mean')(torch.transpose(target_vertices, 0, 1), torch.transpose(output_vertices, 0, 1))

def save_loss_plot(train_losses,val_losses):
    plt.plot(range(len(train_losses)),train_losses,label='Train')
    plt.plot(range(len(val_losses)),val_losses,label='Val')
    plt.title("Loss Plot")
    plt.savefig(os.path.join(root_path,"lossPlot.png"))

def train_model(model, criterion, optimizer, scheduler, cfg):

    best_loss = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []
    unit_diamond_vertices = get_unit_diamond_vertices()
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
            inverse_loss = 0.0
            running_corrects = 0
            ft_all, lbl_all = None, None

            for i, (rock_vertices, targets) in enumerate(data_loader[phrase]):

                optimizer.zero_grad()
                if use_gpu:
                    rock_vertices = Variable(torch.cuda.FloatTensor(rock_vertices.cuda()))
                    targets = Variable(torch.cuda.FloatTensor(targets.cuda()))
                    unit_diamond_vertices = Variable(torch.cuda.FloatTensor(unit_diamond_vertices.cuda()))
                else:
                    rock_vertices = Variable(torch.FloatTensor(rock_vertices))
                    targets = Variable(torch.FloatTensor(targets))
                    unit_diamond_vertices = Variable(torch.FloatTensor(unit_diamond_vertices))
                with torch.set_grad_enabled(phrase == 'train'):
                    eps = 1e-12
                    rock_vertices = torch.reshape(rock_vertices, (-1, 3))
                    outputs = model(rock_vertices)
                    loss = point_wise_L1_loss(outputs, targets, unit_diamond_vertices)
                    inverse_loss += inverse_distance_loss(outputs, rock_vertices)
                    if phrase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * targets.size(0)

            epoch_loss = running_loss / len(data_set[phrase])

            if phrase == 'train':
                print('{} Loss: {:.4f}'.format(phrase, epoch_loss))
                train_losses.append(epoch_loss)
                print("Training inverse distance loss:", inverse_loss/len(data_set[phrase]))
            if phrase == 'val':
                val_losses.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch % 2 == 0:
                    torch.save(copy.deepcopy(model.state_dict()), root_path + '/checkpoints_root/{}.pkl'.format(epoch))

                print('{} Loss: {:.4f}'.format(phrase, epoch_loss))
                print("Validation inverse distance loss:", inverse_loss/len(data_set[phrase]))
        save_loss_plot(train_losses,val_losses)

    return best_model_wts


if __name__ == '__main__':

    model = PointNet(3, 7)
    if use_gpu:
        model.cuda()
    model = nn.DataParallel(model)
    if os.path.isfile(os.path.join(root_path, cfg['ckpt_root'], 'PointNet_best.pkl')):
        model.load_state_dict(torch.load(os.path.join(root_path, cfg['ckpt_root'], 'PointNet_best.pkl')))
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    best_model_wts = train_model(model, criterion, optimizer, scheduler, cfg)
    torch.save(best_model_wts, os.path.join(root_path, cfg['ckpt_root'], 'PointNet_best.pkl'))
