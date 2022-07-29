# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 08:59:25 2022

@author: frank
"""


from ast import Sub
#from types import NoneType
from typing import ForwardRef
# from typing import _ForwardRef as ForwardRef
import torch
import time
import re
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import argparse
import numpy as np
import scipy.io
import requests
import math
from PIL import *
from os import listdir
import os
from glob import glob
from os.path import isfile, isdir, join, splitext
import importlib
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from utils import seed_everything
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime
# from torch.optim.swa_utils import *
from lossfun import BinaryFocalLoss_2
from radam import RAdam
from dataset import TorchDataset
from training_testing_2 import *
import yaml


def train(cfg_path, event, namespace):
    
    with open(cfg_path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))

    # args = parser.parse_args()
    print(args)

    # create log dir
    # args.trained_model = join(args.logdir,args.trained_model )  #c
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    # import model (in models folder, [args.model_name].py)
    model_import_str = 'models.' + args.model_name
    models = importlib.import_module(model_import_str)
    
    # get timestamp
    #   (if resuming from checkpoint, use the timestampe from checkpoint)
    #   (else, use current time)
    now = datetime.now() 
    # convert
    if args.use_pretrained and not args.restart:
        a = args.trained_model.replace('.','_').split('_')
        dt_string = '%s_%s' % (a[-3],a[-2])
    else:
        dt_string = now.strftime("%Y%m%d_%H%M")
    dt_string = now.strftime("%Y%m%d_%H%M")
    args.timestamp = dt_string
    

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device:
        print ("GPU is ready")
    # if use args.half_precision (NOT RECOMMANDED)
    if args.half_precision:
        dtype = torch.float16
    else:
        dtype = torch.float32
    # initialize by RNG by  args.seed
    seed_everything(seed=args.seed)

    # initialize dataset
    train_data = TorchDataset(image_dir = args.train_data,
        repeat=1,augment=True,z_crop=args.z_crop,dtype=dtype,pre_align=args.pre_align,skull_strip=args.skull_strip,
        aug_Rrange=args.aug_Rrange, aug_Trange=args.aug_Trange,
        hflip=args.aug_hflip,vflip=args.aug_vflip)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,num_workers=0,pin_memory=False)
    
    if namespace is not None:
        namespace.dataset_size = len(train_data)  #c

    test_augment = True if args.num_test_aug > 1 else False
    valid_data = TorchDataset(image_dir = args.valid_data,
        repeat=1,augment=test_augment,n_augs=args.num_test_aug,z_crop=args.z_crop,dtype=dtype,pre_align=args.pre_align,skull_strip=args.skull_strip,
        aug_Rrange=args.aug_Rrange, aug_Trange=args.aug_Trange,
        hflip=args.aug_hflip,vflip=args.aug_vflip)        
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False,num_workers=0,pin_memory=False)

    print("Training dataset:\n", train_data)
    print("Testing dataset:\n", valid_data)
    
    print ("Train Start")
    model_kwargs = dict(zip([], []))
    model = models.mymodel(**model_kwargs)
    if args.half_precision:
        model.half()
    
    if args.use_SWA_model:
        global swa_model         
        swa_model = AveragedModel(model,device)
    else:
        swa_model = None


    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of params: %.3fM' % (float(num_params)/1e6))
    if args.use_pretrained:
        # print(namespace.pretrained_path)
        pretrained_state = torch.load(args.trained_model) #c args.trained_model  namespace.pretrained_path
    else:
        pretrained_state = None
    

    ############# calculate class balance weights #########################
    beta = [args.loss_beta, args.loss_beta]
    normal_data_num, detect_data_num = train_data.calculate_slice()
    class_balance_weights = [(1-beta[0])/(1-pow(beta[0],detect_data_num)), (1-beta[1])/(1-pow(beta[1],normal_data_num))]
    alpha = 1/(class_balance_weights[0]+class_balance_weights[1])
    data_weight = [class_balance_weights[0]*alpha, class_balance_weights[1]*alpha] ### class weight for CBFL

    ##### Initializing loss function
    loss_kwargs = dict(zip([], []))
    # eval_str = 'lossfun = ' + args.lossfun
    # exec(eval_str)   
    if 'num_class' not in model_kwargs.keys():
        data_weight = data_weight[1]
    elif model_kwargs['num_class'] == 1:
        data_weight = data_weight[1]
    Loss = BinaryFocalLoss_2(alpha=data_weight,**loss_kwargs)
    
    
    print ("p",data_weight)  
    ##### Initializing optimizer and swa_optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.Learning_rate,weight_decay=args.weight_decay)
    elif args.optimizer == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=args.Learning_rate,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.Learning_rate,weight_decay=args.weight_decay)        
    swa_optimizer = torch.optim.SGD(model.parameters(), lr=args.Learning_rate,weight_decay=args.weight_decay)

    ##### 
    # checkpoint_name =  'pretrain.pt'   #c
    checkpoint_name =  'vol_checkpoint_%s_%s.pt' % ( args.model_name, args.timestamp)
    checkpoint_name = join(args.logdir,checkpoint_name)
    
    ##### Training process
    train_acc , valid_acc, train_loss , valid_loss  = training(args,
            model, train_loader, valid_loader, Loss, data_weight,
            optimizer,args.epoch, device, args.num_class, checkpoint_name, event, namespace,
            class_threshold=args.class_threshold,swa_model=swa_model,pretrained_state=pretrained_state)

    ##### calculate final performance from checkpoint
    if args.epoch > 0:
        checkpoint_state = torch.load(checkpoint_name)
    else:
        checkpoint_state = pretrained_state
    if args.use_SWA_model and checkpoint_state['epoch'] >= args.SWA_start_epoch:
        best_model = swa_model
        
        best_model.load_state_dict(checkpoint_state['swa_dict'])
    else:           
        try:
            #best_model.load_state_dict(checkpoint_state['best_model_wts'])
            #best_model = checkpoint_state['model']
            if isinstance(checkpoint_state['model'], models.mymodel):
                best_model	=	checkpoint_state['model']
            else:
                best_model = models.mymodel(**model_kwargs)
                best_model.load_state_dict(checkpoint_state['model'])
        except:
            best_model = swa_model
            best_model.load_state_dict(checkpoint_state['model'])
    best_model.to(device)
    best_model.eval()
    accuracy, total_loss,pred_all, prob_all, label_all,best_th,best_f1 = evaluate(args,best_model, device, valid_loader,Loss, class_threshold=args.class_threshold)
    test_pred_filename = join(args.logdir,'test_pred_%s_%s.npz' % (args.model_name,args.timestamp))
    np.savez(test_pred_filename ,test_pred = pred_all,test_label = label_all,test_prob=prob_all)
    print ("Accuracy : " , accuracy ,"%")

if __name__=='__main__':
    
    train(cfg_path = 'train_config.yml', event = None, namespace = None)