# -*- coding: utf-8 -*-
"""
Main script for LACE experiments
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pdb
from Utils.Initailize_model import initialize_model
from yaml import load , CLoader as Loader
#Turn off plotting
plt.ioff()


def main(args):
    
    with open(args.yaml , 'r') as stream:
        out = load(stream, Loader=Loader)

    # Name of dataset
    Dataset =out['roboflow']['project']
    
    # Model(s) to be used
    model_name = args.model
    
    # Number of classes in dataset
    num_classes = out['nc']
    
    # Number of runs and/or splits for dataset
    numRuns = out['num_runs']
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Starting Experiments...')
    
    for split in range(0, numRuns):
        #Set same random seed based on split and fairly compare
        torch.manual_seed(split)
        np.random.seed(split)
        np.random.seed(split)
        torch.cuda.manual_seed(split)
        torch.cuda.manual_seed_all(split)
        torch.manual_seed(split)
                

        # Initialize the histogram model for this run
        model_ft, input_size = initialize_model(model_name, num_classes,
                                                feature_extract=args.feature_extraction,
                                                use_pretrained=args.use_pretrained)

        
        # Send the model to GPU if available, use multiple if available
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            list_device= np.arange(0, torch.cuda.device_count(), 1)
            list_device.tolist()
        else:
            list_device = device.type   
        
        model_ft.info()
        value=model_ft.info()
        num_params = value[1]

        if args.feature_extraction:
            mode = 'feature_extract'
        else:
            mode = 'fine_tuning'

        # TRAIN VAL AND TEST USING ULTRALETYCS and store in runs 
        project = '{}/{}/{}/{}'.format(args.folder,
                                            mode,
                                            Dataset,
                                            model_name)
        name='Run_{}'.format(split+1)
        #TRAIN 
        model_ft.train(data=args.yaml, epochs= args.num_epochs, batch=args.train_batch_size ,imgsz=args.resize_size, 
                        device= list_device,pretrained=args.use_pretrained, project = project, name= name,save=args.save_results,
                       seed = split)
        
        #pretrained and freeze already there ( implement later) & dropout

        #  VALIDATE 
        # model_ft.val(save=args.save_results,batch = args.val_batch_size) 
        
        # TEST 
        # model_ft.val(split='test',save=args.save_results)
        # If explicitly want to see the model info such as the number of the parameters and much more.
        #model_ft.info()
        print('**********Run ' + str(split + 1) + model_name + ' Finished**********')
       

def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Losses and Baseline experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments(default: True)')
    parser.add_argument('--folder', type=str, default='path/to/your/Saved_Models/',
                        help='Location to save models')
    parser.add_argument('--yaml', type=str, default='/path/to/your/data.yaml',
                        help='Location to save models')
    parser.add_argument('--feature_extraction', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected/encoder parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=640,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--model', type=str, default='YOLOV8L',
                        help='backbone architecture to use (default: 0.01)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    main(args)
      
