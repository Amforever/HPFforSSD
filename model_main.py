"""
    Author: zhenyuzhang
    All rights reserved.
"""
import argparse
import sys
import os
import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import torch.nn.functional as F    #ByZZY
import time                         #ByZZY
import torch
from torch import nn
from tensorboardX import SummaryWriter
from eval_metrics import compute_eer
import scipy.io as scio
import soundfile as sf
from joblib import Parallel, delayed



from models import  resnext18,resnext34,resnext50,resnext101,resnext152,resnext34_varTanh,resnext50_varTanh,resnext34_varNonBN,resnext50_varNonBN,resnext101_varNonBN,resnext101_varTanh

def compute_mfcc_feats(x):
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta, delta2), axis=0)
    return feats


def diff_layer(input_data, is_diff, is_diff_abs, is_abs_diff, order, direction, name, padding):
    """
    the layer which is used for difference
    :param is_diff: whether make difference or not
    :param is_diff_abs: whether make difference and abs or not
    :param is_abs_diff: whether make abs and difference or not
    :param order: the order of difference
    :param direction: the direction of difference, "inter"(between row) or "intra"(between col)
    :param name: the name of the layer
    :param padding: the method of padding, default is "SAME"
    """

    #print("name: %s, is_diff: %r, is_diff_abs: %r, is_abs_diff: %r, order: %d, direction: %s" % (name, is_diff, is_diff_abs, is_abs_diff, order, direction))

    if order == 0:
        return input_data
    else:
        if order == 1 and direction == "inter":
            filter_diff =np.array([1, -1])
            filter_diff=filter_diff.reshape(1, 1, 2, 1)
            filter_diff_tensor = torch.tensor(filter_diff, dtype=torch.float32)
        elif order == 1 and direction == "intra":
            filter_diff =np.array([1, -1])
            filter_diff=filter_diff.reshape(1, 1, 1, 2)
            filter_diff_tensor = torch.tensor(filter_diff, dtype=torch.float32)
        elif order == 2 and direction == "inter":
            filter_diff =np.array([1, -2, 1])
            filter_diff=filter_diff.reshape(1, 1, 3, 1)
            filter_diff_tensor = torch.tensor(filter_diff, dtype=torch.float32)
        elif order == 2 and direction == "intra":
            filter_diff =np.array([1, -2, 1])
            filter_diff=filter_diff.reshape(1,1, 1, 3)
            filter_diff_tensor = torch.tensor(filter_diff, dtype=torch.float32)
        elif order == 1 and direction == "diagonalzzy":
            filter_diff_tensor = torch.Tensor([[[[0, 1], [-1, 0]]]])
        elif order == 1 and direction == "invdiagonalzzy":
            # b = torch.Tensor([[[[0, 1], [-1, 0]]]])
            filter_diff_tensor =   torch.Tensor([[[[1, 0], [0, -1]]]])
        elif order == 2 and direction == "diagonalzzy":
            filter_diff_tensor =   torch.Tensor([[[[0, 0,1], [0, -2,0],[1,0,0]]]])
        elif order == 2 and direction == "invdiagonalzzy":
            filter_diff_tensor =   torch.Tensor([[[[1, 0,0], [0, -2,0],[0,0,1]]]])


        if is_diff is True:
            output =F.conv2d(input_data,filter_diff_tensor,stride=1,padding=padding)
            return output
        elif is_diff_abs is True:
            output =F.conv2d(input_data,filter_diff_tensor,stride=1,padding=padding)
            output = torch.abs(output)
            return output
        elif is_abs_diff is True:
            input_data = torch.abs(input_data)
            output =F.conv2d(input_data,filter_diff_tensor,stride=1,padding=padding)
            return output
        else:
            return input_data


def compute_SingleMfccFilterHPF_8diff_feats(x):    #compute_Residual_feats
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=72)
    x_matrix=mfcc.reshape(1,1,mfcc.shape[0],mfcc.shape[1])
    # scio.savemat('dataNew.mat', {'XmatrixValue': x_matrix})
                                                                  #is_diff, is_diff_abs, is_abs_diff 表示直接差分，差分后绝对值，绝对之后差分，三者只能有一个是true
    x_matrix_tensor = torch.tensor(x_matrix, dtype=torch.float32)  #(input_data, is_diff, is_diff_abs, is_abs_diff, order, direction, name, padding):
    dif_inter_1 = diff_layer(x_matrix_tensor, True, False, False, 1, "inter", "dif_inter_1", padding=1)
    #dif_inter_1_p=dif_inter_1.resize_(1,1,x.shape[0],x.shape[1])
    dif_inter_1_p = dif_inter_1[:,:,0:69,0:124]
    dif_inter_2 = diff_layer(x_matrix_tensor, True, False, False, 2, "inter", "dif_inter_2", padding=1)
    #dif_inter_2_p = dif_inter_2.resize_(1, 1, x.shape[0], x.shape[1])
    dif_inter_2_p = dif_inter_2[:,:,0:69,0:124]
    dif_intra_1 = diff_layer(x_matrix_tensor, True, False, False, 1, "intra", "dif_intra_1", padding=1)
    #dif_intra_1_p = dif_intra_1.resize_(1, 1, x.shape[0], x.shape[1])
    dif_intra_1_p = dif_intra_1[:,:,0:69,0:124]
    dif_intra_2 = diff_layer(x_matrix_tensor, True, False, False, 2, "intra", "dif_intra_2",padding=1)
    #dif_intra_2_p = dif_intra_2.resize_(1, 1, x.shape[0], x.shape[1])
    dif_intra_2_p = dif_intra_2[:,:,0:69,0:124]
    dif_abs_inter_1 = diff_layer(x_matrix_tensor, True, False, False, 1, "diagonalzzy", "diagonalzzy_1", padding=1)
    #dif_abs_inter_1_p = dif_abs_inter_1.resize_(1, 1, x.shape[0], x.shape[1])
    dif_abs_inter_1_p = dif_abs_inter_1[:,:,0:69,0:124]
    dif_abs_inter_2 = diff_layer(x_matrix_tensor, True, False, False, 2, "diagonalzzy", "diagonalzzy_2", padding=1)
    #dif_abs_inter_2_p = dif_abs_inter_2.resize_(1, 1, x.shape[0], x.shape[1])
    dif_abs_inter_2_p = dif_abs_inter_2[:,:,0:69,0:124]
    dif_abs_intra_1 = diff_layer(x_matrix_tensor, True, False, False, 1, "invdiagonalzzy", "invdiagonalzzy_1", padding=1)
    #dif_abs_intra_1_p = dif_abs_intra_1.resize_(1, 1, x.shape[0], x.shape[1])
    dif_abs_intra_1_p = dif_abs_intra_1[:,:,0:69,0:124]
    dif_abs_intra_2 = diff_layer(x_matrix_tensor, True, False, False, 2, "invdiagonalzzy", "invdiagonalzzy_2", padding=1)
    #dif_abs_intra_2_p = dif_abs_intra_2.resize_(1, 1, x.shape[0], x.shape[1])
    dif_abs_intra_2_p = dif_abs_intra_2[:,:,0:69,0:124]
    feats4Dim=torch.cat((dif_inter_1_p, dif_inter_2_p, dif_intra_1_p, dif_intra_2_p, dif_abs_inter_1_p, dif_abs_inter_2_p, dif_abs_intra_1_p, dif_abs_intra_2_p),dim = 1)
    feats = torch.squeeze(feats4Dim, 0)#将第一维删去，得到channel hight width
    # featsnumpy = feats.numpy()
    # scio.savemat('dataRes8DNew.mat', {'Xfeats8DValue': featsnumpy})
    #feats=feats4Dim.view(feats4Dim.shape[1]*feats4Dim.shape[2],feats4Dim.shape[3])  #30*508按照列拼接的形式作为一个大矩阵。
    return feats