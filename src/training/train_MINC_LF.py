# Reproduce of material recongnition paper "A 4D light-field dataset and CNN architectures for material recognition", Wang, Ting Chun
# ZhuJun Yan, Hiroaki, Ebi Chandraker, ManmohanEfros, Alexei A.Ramamoorthi, Ravi

from __future__ import print_function

######################################################
"""
REQUIREMENTS:
matplotlib==2.2.2
simplejson==3.16.0
numpy==1.14.1
opencv_python==3.4.3.18
horovod==0.13.5
photutils==0.5
scipy==1.1.0
torch==0.4.0
pyquaternion==0.9.2
tqdm==4.25.0
pyrr==0.9.2
Pillow==5.2.0
torchvision==0.2.1
PyYAML==3.13
"""

######################################################


import argparse
import ConfigParser
import os
import random
import shutil
import numpy as np 
import logging
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.models as models
import datetime
import json
import glob
import os 
import copy

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageEnhance

from math import acos
from math import sqrt
from math import pi    

from os.path import exists, basename
from os.path import join

import cv2
import colorsys,math

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
logger = logging.getLogger(__name__)

##################################################
# NEURAL NETWORK MODEL
##################################################

layer_dic = {
    "0" : "conv1_1",
    "2" : "conv1_2",
    "5" : "conv2_1",
    "7" : "conv2_2",
    "10" : "conv3_1",
    "12" : "conv3_2",
    "14" : "conv3_3",
    "17" : "conv4_1",
    "19" : "conv4_2",
    "21" : "conv4_3",
    "24" : "conv5_1",
    "26" : "conv5_2",
    "28" : "conv5_3"

}


class LF_MINC(nn.Module):
    def __init__(
            self,
            pretrained=False,
            lf_channel=64,
            lf_kernel_size=7,
            minc_path='/opt/project/pretrain_model/minc_vgg16.npy',
            class_num = 12
        ):
        super(LF_MINC, self).__init__()

        self.vgg_full = models.vgg16(pretrained=False).features
        
        self.load_pretrian_MINC(minc_path)
        self.vgg = nn.Sequential()
        # angular filter layer
        conv_temp = nn.Conv2d(3,lf_channel,kernel_size=lf_kernel_size, stride=lf_kernel_size,padding=0)
        torch.nn.init.xavier_uniform(conv_temp.weight)
        self.vgg.add_module(str(0), conv_temp)
        self.vgg.add_module(str(1),nn.ReLU(inplace=True))

        # change the first VGG layer
        conv_temp = nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1)
        torch.nn.init.xavier_uniform(conv_temp.weight)
        self.vgg.add_module(str(2),conv_temp)
        #load vgg 16 layer
        for i_layer in range(len(self.vgg_full._modules)):
            if i_layer == 0:
                continue
            self.vgg.add_module(str(i_layer+2), self.vgg_full[i_layer])

        #add classification layer
        # current_layer_id = len(self.vgg_full._modules) + 2
        self.classifier = models.vgg16(pretrained=False).classifier
        classifier_temp = list(self.classifier.children())[:-1] # Remove last layer
        classifier_temp.extend([nn.Linear(4096, class_num)]) # Add our layer with 12 outputs
        self.classifier = nn.Sequential(*classifier_temp) # Replace the model classifier

        # self.classifer.add_module(str(current_layer_id), nn.Linear(512 * 7 * 7,4096))
        # self.classifer.add_module(str(current_layer_id+1),nn.ReLU(inplace=True))
        # self.classifer.add_module(str(current_layer_id+2), nn.Linear(4096,4096))
        # self.classifer.add_module(str(current_layer_id+3),nn.ReLU(inplace=True))
        # self.classifer.add_module(str(current_layer_id+4), nn.Linear(4096,class_num))
        print("model init with MINC pretrain done")




    def load_pretrian_MINC(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.vgg_full, data)




    def forward(self, x):
        '''Runs inference on the neural network'''
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

       
                        
def load_module_npy(module, data):
    global layer_dic
    for name, child in module._modules.items():
        if name in layer_dic:
            if layer_dic[name] in data:
                # logging.info("Loading {} => {}".format(layer_dic[name], child))
                print("load " + layer_dic[name] + " layer")
                weight_shape = tuple(child.weight.size())
                weights = data[layer_dic[name]]['weights']
                if weight_shape != weights.shape:
                    print("reshape weights at layer " + layer_dic[name])
                    # logging.info("\tReshaping weight {} => {}"
                    #       .format(weights.shape, weight_shape))
                    weights = weights.reshape(weight_shape)
                weights = torch.from_numpy(weights)
                bias = data[layer_dic[name]]['biases']
                bias = torch.from_numpy(bias)
                child.weight.data.copy_(weights)
                child.bias.data.copy_(bias)


def loadimages(root, format):
    """
    Find all the images in the path and folders, return them in imgs. 
    """
    imgs = []

    def add_img_files(path):
        for imgpath in glob.glob(path+"/*."+format):
            if exists(imgpath):
                imgs.append(imgpath)

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) 
                        if os.path.isdir(os.path.join(path,o))]
        if len(folders)>0:
            for path_entry in folders:                
                explore(path_entry)
        else:
            add_img_files(path)

    explore(root)
    return imgs

class Loadpatch(data.Dataset):
    def __init__(self,root,img_format='png',transform=None, num_class=12):
        self.root = root
        self.imgs = []
        self.img_format = img_format
        self.transform = transform
        self.num_class = num_class
        # def load_data(path):
        #         '''Recursively load the data.  This is useful to load all of the FAT dataset.'''
        #         imgs = loadimages(path)
        #
        #         # Check all the folders in path
        #         for name in os.listdir(str(path)):
        #             imgs += loadimages(path +"/"+name, self.img_format)
        #         return imgs
        #
        def load_data_fromcsv(path):
            train_info = pd.read_csv(path+'train.csv')
            return train_info
        
        self.imgs = load_data_fromcsv(self.root)
        # np.random.shuffle(self.imgs)
    def __len__(self):
        # When limiting the number of data

        return len(self.imgs)   

    def __getitem__(self, index):
        
        img_path = self.imgs.iloc[index,0].replace('/media/logan/data/','/opt/')
        image = Image.open(img_path).convert('RGB')
        label = np.array([self.imgs.iloc[index,1].astype('int')-1])
        # label = np.zeros(self.num_class).astype(float)
        # label[label_temp] = 1.0
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        totensor = transforms.Compose([transforms.ToTensor()])
        return {'image': totensor(image),
                'label': torch.from_numpy(label)}

conf_parser = argparse.ArgumentParser(
    description=__doc__, # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False
    )
parser = argparse.ArgumentParser()
parser.add_argument('--lr', 
    type=float, 
    default=0.0001, 
    help='learning rate, default=0.001')

parser.add_argument('--root', 
    default="/opt/data/LF_dataset_material_recong/patch_image/",
    help='learning rate, default=0.001')

parser.add_argument('--gpuids',
    nargs='+', 
    type=int, 
    default=[0], 
    help='GPUs to use')
parser.add_argument('--epochs', 
    type=int, 
    default=60,
    help="number of epochs to train")
parser.add_argument('--outf', 
    default='tmp', 
    help='folder to output images and model checkpoints, it will \
    add a train_ in front of the name')
parser.add_argument('--batchsize', 
    type=int, 
    default=32, 
    help='input batch size')
# Read the config but do not overwrite the args written 
args, remaining_argv = conf_parser.parse_known_args()
defaults = { "option":"default" }

parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)


net = LF_MINC().cuda()
net = torch.nn.DataParallel(net,opt.gpuids).cuda()

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters,lr=opt.lr)

criterion = nn.CrossEntropyLoss()
nb_update_network = 0

train_dataset = Loadpatch(root=opt.root, transform=transforms.Compose([ToTensor()]))
trainingdata = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                        shuffle=True)

nb_update_network = 0
def _runnetwork(epoch, loader, train=True):
    global nb_update_network
    # net
    if train:
        net.train()
    else:
        net.eval()

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, targets in enumerate(loader):

        

        data = Variable(targets['image'].cuda())
        
        output_label = net(data)
        _, preds = torch.max(output_label, 1)

        if train:
            optimizer.zero_grad()
        target_label = Variable(targets['label'].cuda())
        target_label.squeeze_()

        loss = None
        
        # loss computation
        loss = criterion(output_label, target_label) 


        if train:
            loss.backward()
            optimizer.step()
            nb_update_network+=1

        # if train:
        #     namefile = '/loss_train.csv'
        # else:
        #     namefile = '/loss_test.csv'

        # with open (opt.outf+namefile,'a') as file:
        #     s = '{}, {},{:.15f}\n'.format(
        #         epoch,batch_idx,loss.data[0]) 
        #     # print (s)
        #     file.write(s)
        running_loss += loss.item() * data.size(0)
        running_corrects += torch.sum(preds == target_label.data)
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, epoch_loss, epoch_acc))
        # if train:
        #     if batch_idx % opt.loginterval == 0:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
        #             epoch, batch_idx * len(data), len(loader.dataset),
        #             100. * batch_idx / len(loader), loss.data[0]))
        # else:
        #     if batch_idx % opt.loginterval == 0:
        #         print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
        #             epoch, batch_idx * len(data), len(loader.dataset),
        #             100. * batch_idx / len(loader), loss.data[0]))

        # # break
        # if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        #     torch.save(net.state_dict(), '{}/net_{}.pth'.format(opt.outf, opt.namefile))
        #     break


for epoch in range(1, opt.epochs + 1):

    # if not trainingdata is None:
    _runnetwork(epoch,trainingdata)

    # if not opt.datatest == "":
    #     _runnetwork(epoch,testingdata,train = False)
    #     if opt.data == "":
    #         break # lets get out of this if we are only testing
    # try:
    #     torch.save(net.state_dict(), '{}/net_{}_{}.pth'.format(opt.outf, opt.namefile ,epoch))
    # except:
    #     pass

    # if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
    #     break
