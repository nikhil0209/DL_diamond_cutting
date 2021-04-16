import copy
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import get_train_config
from data import ModelNet40
from models import MeshNet
from utils import append_feature, calculate_map


cfg = get_train_config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
use_gpu = torch.cuda.is_available()

data_set = {
    x: ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
}
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=False)
    for x in ['train', 'test']
}


def train_model(model, criterion, optimizer, scheduler, cfg):

    best_loss = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        for phrase in ['train', 'test']:

            if phrase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            ft_all, lbl_all = None, None

            for i, (centers, corners, normals, neighbor_index, targets, impurity_label) in enumerate(data_loader[phrase]):

                optimizer.zero_grad()
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

                with torch.set_grad_enabled(phrase == 'train'):
                    outputs, feas = model(centers, corners, normals, neighbor_index, impurity_label)
                    loss = criterion(outputs, targets)

                    if phrase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phrase == 'test':
                        ft_all = append_feature(ft_all, feas.detach())
                        lbl_all = append_feature(lbl_all, targets.detach(), flaten=True)

                    running_loss += loss.item() * centers.size(0)

            epoch_loss = running_loss / len(data_set[phrase])

            if phrase == 'train':
                print('{} Loss: {:.4f}'.format(phrase, epoch_loss))

            if phrase == 'test':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch % 2 == 0:
                    torch.save(copy.deepcopy(model.state_dict()), 'ckpt_root/{}.pkl'.format(epoch))

                print('{} Loss: {:.4f}'.format(phrase, epoch_loss))

    return best_model_wts


if __name__ == '__main__':

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    if use_gpu:
        model.cuda()
    model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()#TODO switch to L1 after a few epochs when it becomes small enough?
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])

    best_model_wts = train_model(model, criterion, optimizer, scheduler, cfg)
    torch.save(best_model_wts, os.path.join(cfg['ckpt_root'], 'MeshNet_best.pkl'))
