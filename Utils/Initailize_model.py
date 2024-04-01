# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy
from ultralytics import YOLO, RTDETR
## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models
import pdb

def initialize_model(model_name, num_classes,feature_extract=False,
                     use_pretrained=False):
    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
  
    #Select backbone architecture
    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet50_wide":
        model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet50_next":
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "densenet121":
        
        model_ft = models.densenet121(pretrained=use_pretrained,memory_efficient=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential()
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "YOLOV8N":
        
        # model_ft = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model_ft = YOLO('yolov8n.yaml')
        #pdb.set_trace()
        input_size = 512

    elif model_name == "YOLOV8M":
        
        # model_ft = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model_ft = YOLO('yolov8m.yaml')
        input_size = 512    


    elif model_name == "YOLOV8L":
        
        # model_ft = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model_ft = YOLO('yolov8l.yaml')
        input_size = 512  

    elif model_name == "rtdetr":
        
        # model_ft = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model_ft = RTDETR('/scratch/user/yashzambre/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
        input_size = 512        

    elif model_name == "convnext":
        model_ft = models.convnext_base(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(model_ft.classifier[2].in_features, num_classes)
        input_size = 224
    
    else:
        raise RuntimeError('{} not implemented'.format(model_name))
    

    return model_ft, input_size

