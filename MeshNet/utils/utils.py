import os
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

def get_unit_diamond_vertices(root_path):
    vertices = np.transpose(np.genfromtxt(root_path + '/data/unit_diamond.txt', delimiter=' '))
    return torch.from_numpy(vertices).float()

def save_loss_plot(train_losses,val_losses,root_path):
    plt.plot(range(len(train_losses)),train_losses,label='Train')
    plt.plot(range(len(val_losses)),val_losses,label='Val')
    plt.title("Loss Plot")
    plt.savefig(os.path.join(root_path,"lossPlot.png"))

def stochastic_loss( outputs, targets):
    scale_actual = targets[:, -1]
    scale_predicted = outputs[:, -1]
    scale_loss = torch.log(torch.maximum(torch.div(scale_actual, scale_predicted), torch.div(scale_predicted, scale_actual)))
    r_x_actual = targets[:, 3]
    r_x_predicted = outputs[:, 3]
    r_x_loss = 1 - torch.cos(torch.sub(r_x_actual, r_x_predicted))
    r_y_actual = targets[:, 4]
    r_y_predicted = outputs[:, 4]
    r_y_loss = 1 - torch.cos(torch.sub(r_y_actual, r_y_predicted))
    r_z_actual = targets[:, 5]
    r_z_predicted = outputs[:, 5]
    r_z_loss = 1 - torch.cos(torch.sub(r_z_actual, r_z_predicted))
    translation_loss = nn.L1Loss()(outputs[:,0:3], targets[:,0:3])
    loss = scale_loss + r_x_loss + r_y_loss + r_z_loss + translation_loss
    return loss

def point_wise_L1_loss(outputs, targets, unit_diamond_vertices):
    target_rotations = [R.from_euler('xyz', [[x[3], x[4], x[5]]]).as_matrix() for x in targets]
    predicted_rotations = [R.from_euler('xyz', [[x[3], x[4], x[5]]]).as_matrix() for x in outputs]
    target_rotations = torch.tensor(target_rotations).squeeze(1).to(outputs)
    predicted_rotations = torch.tensor(predicted_rotations).squeeze(1).to(outputs)

    target_vertices=torch.mul(targets[:, -1][:,None,None],torch.matmul(target_rotations, unit_diamond_vertices))+targets[:, :3].unsqueeze(-1)
    output_vertices=torch.mul(outputs[:, -1][:,None,None],torch.matmul(predicted_rotations, unit_diamond_vertices))+outputs[:, :3].unsqueeze(-1)

    loss = nn.MSELoss(reduction='none')(torch.transpose(target_vertices, 1, 2), torch.transpose(output_vertices, 1, 2))
    loss = torch.mean(torch.mean(torch.sum(loss,-1),-1),-1)
    return loss

def axis_aligned_miou(outputs, target):
    center_o = outputs[:,:3]
    scale_o = outputs[:,-1:]
    center_t = target[:,:3]
    scale_t = target[:,-1:]


    intersection_c1 = torch.max(center_o-scale_o,center_t-scale_t)
    intersection_c2 = torch.min(center_o+scale_o,center_t+scale_t)
    intersection = torch.max(torch.zeros(intersection_c1.shape),intersection_c2 - intersection_c1)
    intersection = torch.prod(intersection,-1)
    union = torch.pow(scale_o*2,3)+torch.pow(scale_t*2,3)
    union = union.squeeze(-1)-intersection
    iou = torch.div(intersection,union+1e-5)
    miou = torch.mean(iou)
    return miou
    


