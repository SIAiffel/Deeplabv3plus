import network
import utils
import os
import random
import argparse
import numpy as np

#import Satellites 추가
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torch.onnx
import sys
#import torchvision.transforms as T
import cv2
import torchvision
import time
import torchvision.transforms as transforms

import torch.nn.functional as F

# https://discuss.pytorch.org/t/custom-ensemble-approach/52024

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        #self.modelA.fc = nn.Identity()
                
        #Now We  a head on top of those models
        # The trained model I am passing here is binary type based on densenet169
        # so I added a sequential `nn.Linear(num_ftrs, 2)` with them for output when they were trained
        # so, sould I use 2 instead of 1664 ? but used 1664 any way. 
        # Create new classifier

        # 256*512*512 -> 3*512*512
        # 
        #self.classifier = nn.Linear(1572864, 3) 
        #self.classifier = nn.Conv2d(in_channels=, out_channels=, kernel_size=1, padding=0, stride=1)  #??
        self.classifier = nn.Sequential(    
            nn.Conv2d(6, 3, 1))
        #)
        
    def forward(self, x1, x2):
        #with torch.no_grad():
        x1 = self.modelA(x1)  # clone to make sure x is not changed by inplace methods
        #x1 = x1.view(x1.size(0), -1)
        #with torch.no_grad():
        x2 = self.modelB(x2)
        #x2 = x2.view(x2.size(0), -1)        
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        #print(x.shape)
        #print(x1.shape)
        #print(x2.shape)
        return x #self.classifier( torch.cat( [ x1, x2 ], dim=1 ) )


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/SIA/buildings',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='satellites',
                        choices=['voc', 'cityscapes', 'satellites'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=3,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")                    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt_second", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    parser.add_argument('-i', '--input', help='path to input video')


    return parser


def main():
    opts = get_argparser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Set up model
    model_map = {
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    modelA = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(modelA.classifier)
    utils.set_bn_momentum(modelA.backbone, momentum=0.01)

    modelB = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(modelB.classifier)
    utils.set_bn_momentum(modelB.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': modelA.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': modelA.classifier.parameters(), 'lr': opts.lr},
        {'params': modelB.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': modelB.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            modelA.load_state_dict(checkpoint["model_state"])
            #print("check")

            if opts.ckpt_second is not None and os.path.isfile(opts.ckpt_second):
                checkpoint_2 = torch.load(opts.ckpt_second, map_location=torch.device('cpu'))
                modelB.load_state_dict(checkpoint_2["model_state"])
                #for param in modelA.parameters():
                #    param.requires_grad_(False)

                #for param in modelB.parameters():
                #    param.requires_grad_(False)

                #model = MyEnsemble(modelA, modelB)
                modelA = nn.DataParallel(modelA)
                modelB = nn.DataParallel(modelB)
                modelA.to(device)    
                modelB.to(device)    
                print("clear")

            else:
                model = nn.DataParallel(modelA)                
                model.to(device)    
                print("check")    
            if opts.continue_training:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                cur_itrs = checkpoint["cur_itrs"]
                best_score = checkpoint['best_score']
                print("Training state restored from %s" % opts.ckpt)
            print("Model restored from %s" % opts.ckpt)
            print("Model restored from %s" % opts.ckpt_second)
            del checkpoint  # free memory
            
    else:
        print("[!] Retrain")
        model = nn.DataParallel(modelA)
        model.to(device)
     
    modelA.eval()
    modelB.eval()
    #segment(model, './images/example_05.png')

    cap = cv2.VideoCapture(opts.input)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    # get the frame width and height
    #cap.set(3, 640)
    #cap.set(4, 800)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))


    #save_name = f"{opts.input.split('/')[-1].split('.')[0]}"
    # define codec and create VideoWriter object 
    #out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
    #                    cv2.VideoWriter_fourcc(*'mp4v'), 30, 
    #                    (frame_width, frame_height))

    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))
    ax2.set_title("Segmentation")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    im1 = ax1.imshow(grab_frame(cap))
    im2 = ax2.imshow(grab_frame(cap))


    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:
            # get the start time
            start_time = time.time()
            with torch.no_grad():
                frame = cv2.resize(frame,(512,512),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                # get predictions for the current frame
                outputs_A = get_segment_labels_A(frame, modelA, device)
                outputs_B = get_segment_labels_B(frame, modelB, device)

            # draw boxes and show current frame on screen
            segmented_image = draw_segmentation_map(outputs_A[0], outputs_B[0])

            final_image = image_overlay(frame, segmented_image)
            # get the end time
            end_time = time.time()
            # get the fps
            fps = 1 / (end_time - start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            # press `q` to exit
            wait_time = max(1, int(fps/4))

            #cv2.imshow('image', final_image)
            #out.write(final_image)
            avg_fps = total_fps / frame_count
            str = "FPS : %0.1f" % fps
            #print(f"Average FPS: {avg_fps:.3f}")
            cv2.putText(final_image, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 250, 255), 2)
            
            im1.set_data(frame)
            im2.set_data(final_image)
            plt.pause(0.01)

            #if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            #    break
        else:
            break

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    


def get_segment_labels_A(image, modelA, device):
    # transform the image to tensor and load into computation device
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs_A = modelA(image)
    # uncomment the following lines for more info
    # print(type(outputs))
    # print(outputs['out'].shape)
    # print(outputs)
    return outputs_A

def get_segment_labels_B(image, modelB, device):
    # transform the image to tensor and load into computation device
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs_B = modelB(image)
    # uncomment the following lines for more info
    # print(type(outputs))
    # print(outputs['out'].shape)
    # print(outputs)
    return outputs_B


def draw_segmentation_map(outputs_A, outputs_B):
    labels_A = torch.argmax(outputs_A.squeeze(), dim=0).detach().cpu().numpy()
    labels_B = torch.argmax(outputs_B.squeeze(), dim=0).detach().cpu().numpy()

    label_colors = np.array([(0, 0, 0),(254, 94, 0), (128, 64, 128)])  # 0=background 1=buliding, 2=road

    red_map_A = np.zeros_like(labels_A).astype(np.uint8)
    green_map_A = np.zeros_like(labels_A).astype(np.uint8)
    blue_map_A = np.zeros_like(labels_A).astype(np.uint8)
    
    red_map_B = np.zeros_like(labels_A).astype(np.uint8)
    green_map_B = np.zeros_like(labels_A).astype(np.uint8)
    blue_map_B = np.zeros_like(labels_A).astype(np.uint8)

    
    for label_num in range(0, 3):
        index = labels_A == label_num
        red_map_A[index] = label_colors[label_num, 0]
        green_map_A[index] = label_colors[label_num, 1]
        blue_map_A[index] = label_colors[label_num, 2]

    for label_num in range(0, 3):
        index = labels_B == label_num
        red_map_B[index] = label_colors[label_num, 0]
        green_map_B[index] = label_colors[label_num, 1]
        blue_map_B[index] = label_colors[label_num, 2]

        
    segmented_image = np.stack([red_map_A, green_map_A, blue_map_B], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):
    alpha = 0.6 # how much transparency to apply
    beta = 1 - alpha # alpha + beta should equal 1
    gamma = 0 # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)

    return image


def grab_frame(cap):
  _, frame = cap.read()
  return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    main()