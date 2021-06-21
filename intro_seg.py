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
import torchvision.transforms as T
import cv2
import torchvision


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
    parser.add_argument("--ckpt", default='./checkpoints/best_deeplabv3plus_mobilenet_satellites_os16.pth', type=str,
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

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
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

    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)        

    model.eval()

    #segment(model, './images/example_05.png')

    BLUR = True           # 원본 이미지 블러 처리하기 
    #BG_PTH = "emoji.jpg"  # 백그라운드 이미지 입히기 
    #bg_image = cv2.imread(BG_PTH)
    #bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

    blur_value = (51, 51)
    # Start a video cam session
    #video_session = cv2.VideoCapture(0)
    video_session = cv2.VideoCapture('./egypt.mp4')   # 영상 주소 넣기

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))
    ax2.set_title("Segmentation")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Create two image objects to picture on top of the axes defined above
    im1 = ax1.imshow(grab_frame(video_session))
    im2 = ax2.imshow(grab_frame(video_session))

    # Switch on the interactive mode in matplotlib
    plt.ion()
    plt.show()

    # Read frames from the video, make realtime predictions and display the same
    while True:
        frame = grab_frame(video_session)

        # Ensure there's something in the image (not completely blacnk)
        if np.any(frame):

            # Read the frame's width, height, channels and get the labels' predictions from utilities
            width, height, channels = frame.shape
            labels = get_pred(frame, model)
            
            if BLUR:
                # Wherever there's empty space/no person, the label is zero 
                # Hence identify such areas and create a mask (replicate it across RGB channels)
                mask = labels == 0
                mask = np.repeat(mask[:, :, np.newaxis], channels, axis = 2)

                # Apply the Gaussian blur for background with the kernel size specified in constants above
                blur = cv2.GaussianBlur(frame, blur_value, 0)
                #frame[mask] = blur[mask]
                ax1.set_title("Original Video")
            else:
                # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
                # Hence wherever person is predicted, the label returned will be 15
                # Subsequently repeat the mask across RGB channels 
                mask = labels == 15
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
                
                # Resize the image as per the frame capture size
                #bg = cv2.resize(bg_image, (height, width))
                #bg[mask] = frame[mask]
                #frame = bg
                #ax1.set_title("Background Changed Video")
            
            # Set the data of the two images to frame and mask values respectively
            im1.set_data(frame)
            im2.set_data(mask * 255)
            plt.pause(0.01)
            
        else:
            break

    # Empty the cache and switch off the interactive mode
    torch.cuda.empty_cache()
    plt.ioff()


    

def crop(image, source, nc=2):
  """
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  """
  label_colors = np.array([(0, 0, 0),(70, 70, 70), (24, 26, 100)])  # 0=background 1=buliding, 2=road

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)

  foreground=cv2.imread(source)
  background_ori=cv2.imread(source)

  foreground=cv2.cvtColor(foreground,cv2.COLOR_BGR2RGB)
  foreground=cv2.resize(foreground,(r.shape[1],r.shape[0]))
  
  background=255*np.ones_like(rgb).astype(np.uint8)

  foreground=foreground.astype(float)
  background=background.astype(float)
  
  th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)
  alpha = cv2.GaussianBlur(alpha, (7,7),0)
  alpha=alpha.astype(float)/255

  #foreground=cv2.multiply(alpha,foreground)

  #background=cv2.multiply(1.0-alpha,background)

  outImage=cv2.add(alpha,background_ori)

  return outImage/255



# Define the helper function
def decode_segmap(image, nc=2):
  """
    label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  """

  label_colors = np.array([(0, 0, 0),(70, 70, 70), (24, 26, 100)])  # 0=background 1=buliding, 2=road

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb




def segment(net, path, show_orig=True, dev='cuda'):
  img = Image.open(path)
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  # Comment the Resize and CenterCrop for better inference results
  trf = T.Compose([T.Resize(640), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  #rgb = decode_segmap(om)
  rgb = crop(om, path)

  plt.imshow(rgb); plt.axis('off'); plt.show()


# Given a video capture object, read frames from the same and convert it to RGB
def grab_frame(cap):
  _, frame = cap.read()
  return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def get_pred(img, model):
  # See if GPU is available and if yes, use it
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Define the standard transforms that need to be done at inference time
  imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
  preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                                std  = imagenet_stats[1])])
  input_tensor = preprocess(img).unsqueeze(0)
  input_tensor = input_tensor.to(device)

  # Make the predictions for labels across the image
  with torch.no_grad():
      output = model(input_tensor)[0]
      output = output.argmax(0)

  # Return the predictions
  return output.cpu().numpy()



if __name__ == '__main__':
    main()
