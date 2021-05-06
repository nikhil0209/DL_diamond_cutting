import numpy as np
import torch
import os
from matplotlib import pyplot as plt
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

def get_unit_diamond_vertices(root_path):
    vertices = np.transpose(np.genfromtxt(root_path + '/data/unit_diamond.txt', delimiter=' '))
    return torch.from_numpy(vertices).float()

def save_loss_plot(train_losses,val_losses,root_path):
    plt.plot(range(len(train_losses)),train_losses,label='Train')
    plt.plot(range(len(val_losses)),val_losses,label='Val')
    plt.title("Loss Plot")
    plt.savefig(os.path.join(root_path,"lossPlot.png"))

def regression_classification_loss(center_probs, rot_scale, center_points, target_scale_and_rotation, alpha = 0.5):
    #binary cross entropy loss for center predictions
    #target vector
    target = torch.zeros(center_probs.squeeze(1).shape).to(center_probs)
    for i, x in enumerate(center_points):
        target[i,x[0],x[1],x[2]]=1
    center_probs_lin = center_probs.reshape(center_probs.shape[0],-1)
    target_lin = target.reshape(target.shape[0],-1)
    target_lin_indexes = torch.nonzero(target_lin)[:,1]
    center_loss_all = torch.nn.CrossEntropyLoss()(center_probs_lin,target_lin_indexes)
    #above will average over all. Very skewed towards negative examples. So add loss for positive example again
    center_probs_lin_ll = torch.log_softmax(center_probs_lin,-1)
    center_loss = - torch.sum(torch.mul(center_probs_lin_ll,target_lin))/center_probs.shape[0]
    center_loss += center_loss_all

    #L1 loss for rotation and scale predictions
    rot_scale_loss = torch.nn.L1Loss()(rot_scale, target_scale_and_rotation)
    return alpha * center_loss + (1-alpha) * rot_scale_loss

def axis_aligned_miou(outputs, target):
    center_o = outputs[:,:3]
    scale_o = outputs[:,-1:]
    center_t = target[:,:3]
    scale_t = target[:,-1:]

    intersection_c1 = torch.max(center_o-scale_o,center_t-scale_t)
    intersection_c2 = torch.min(center_o+scale_o,center_t+scale_t)
    intersection = torch.max(torch.zeros(intersection_c1.shape).to(outputs),intersection_c2 - intersection_c1)
    intersection = torch.prod(intersection,-1)
    union = torch.pow(scale_o*2,3)+torch.pow(scale_t*2,3)
    union = union.squeeze(-1)-intersection
    iou = torch.div(intersection,union+1e-5)
    miou = torch.mean(iou)
    return miou
    
def point_wise_L1_loss(outputs, targets, unit_diamond_vertices):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = torch.tensor(0.0).to(device)
    for i in range(outputs.shape[0]):
        r_target = torch.from_numpy(R.from_euler('xyz', [[targets[i, 3], targets[i, 4], targets[i, 5]]]).as_matrix()).float().to(device)
        target_vertices = torch.mul(targets[i, -1], (torch.matmul(r_target, unit_diamond_vertices))) + torch.reshape(targets[i, 0:3], (3, 1))
        r_output = torch.from_numpy(R.from_euler('xyz', [[outputs[i, 3], outputs[i, 4], outputs[i, 5]]]).as_matrix()).float().to(device)
        output_vertices = torch.mul(outputs[i, -1], (torch.matmul(r_output, unit_diamond_vertices))) + torch.reshape(outputs[i, 0:3], (3, 1))
        loss += nn.L1Loss(reduction='mean')(torch.transpose(target_vertices, 0, 1), torch.transpose(output_vertices, 0, 1))
    loss /= outputs.shape[0]
    return loss


def point_wise_mse_loss(outputs, targets, unit_diamond_vertices):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    r_target = torch.from_numpy(R.from_euler('xyz', [[targets[0, 3], targets[0, 4], targets[0, 5]]]).as_matrix()).float().to(device)
    target_vertices = torch.mul(targets[0, -1], (torch.matmul(r_target, unit_diamond_vertices))) + torch.reshape(targets[0, 0:3], (3, 1))
    r_output = torch.from_numpy(R.from_euler('xyz', [[outputs[0, 3], outputs[0, 4], outputs[0, 5]]]).as_matrix()).float().to(device)
    output_vertices = torch.mul(outputs[0, -1], (torch.matmul(r_output, unit_diamond_vertices))) + torch.reshape(outputs[0, 0:3], (3, 1))
    return nn.MSELoss(reduction='mean')(torch.transpose(target_vertices, 0, 1), torch.transpose(output_vertices, 0, 1))

def get_output_from_prediction(center_probs, pred_rot_scale, pitch, radius):
    center_vertices = []
    #only works for batch=1
    center_probs_lin = center_probs.reshape(center_probs.shape[0],-1)
    center_probs_lin = nn.functional.softmax(center_probs_lin,-1)
    center_idx = center_probs_lin==torch.max(center_probs_lin)
    center_idx = center_idx.reshape(center_probs.shape)
    center_idx = torch.nonzero(center_idx)[0][2:]
    #convert center_idx to a coordinate using center of that voxel
    center = (center_idx-radius)*pitch

    outputs = torch.cat((center.unsqueeze(0).to(pred_rot_scale),pred_rot_scale),dim=-1)
    return outputs