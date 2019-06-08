#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function
import yaml
import sys 
import fnmatch
import os

import numpy as np
import cv2

# import rospy
# import rospkg
# from std_msgs.msg import String, Empty
# from cv_bridge import CvBridge, CvBridgeError
# from sensor_msgs.msg import Image as ImageSensor_msg
# from geometry_msgs.msg import PoseStamped

from PIL import Image
from PIL import ImageDraw

import torch
import torchvision.transforms as transforms
import math
# Import DOPE code
# rospack = rospkg.RosPack()
# g_path2package = rospack.get_path('dope')
g_path2package = '/opt/project'
sys.path.append("{}/src/inference".format(g_path2package))
from cuboid import *
from detector import *

### Global Variables
# g_bridge = CvBridge()
g_img = None
g_draw = None


### Basic functions
# def __image_callback(msg):
#     '''Image callback'''
#     global g_img
#     g_img = g_bridge.imgmsg_to_cv2(msg, "rgb8")
    # cv2.imwrite('img.png', cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))  # for debugging


### Code to visualize the neural network output

def DrawLine(point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    global g_draw
    if not point1 is None and point2 is not None:
        g_draw.line([point1,point2], fill=lineColor, width=lineWidth)

def DrawDot(point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    global g_draw
    if point is not None:
        xy = [
            point[0]-pointRadius, 
            point[1]-pointRadius, 
            point[0]+pointRadius, 
            point[1]+pointRadius
        ]
        g_draw.ellipse(xy, 
            fill=pointColor, 
            outline=pointColor
        )

def DrawCube(points, color=(255, 0, 0)):
    '''
    Draws cube with a thick solid line across 
    the front top edge and an X on the top face.
    '''

    lineWidthForDrawing = 2

    # draw front
    DrawLine(points[0], points[1], color, lineWidthForDrawing)
    DrawLine(points[1], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[0], color, lineWidthForDrawing)
    
    # draw back
    DrawLine(points[4], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[7], color, lineWidthForDrawing)
    DrawLine(points[4], points[7], color, lineWidthForDrawing)
    
    # draw sides
    DrawLine(points[0], points[4], color, lineWidthForDrawing)
    DrawLine(points[7], points[3], color, lineWidthForDrawing)
    DrawLine(points[5], points[1], color, lineWidthForDrawing)
    DrawLine(points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(points[0], pointColor=color, pointRadius = 4)
    DrawDot(points[1], pointColor=color, pointRadius = 4)

    # draw x on the top 
    DrawLine(points[0], points[5], color, lineWidthForDrawing)
    DrawLine(points[1], points[4], color, lineWidthForDrawing)
def OverlayBeliefOnImage(img, beliefs, name, path="", factor=0.7, grid=3,
                         norm_belief=True):
    """ python
    take as input
    img: a tensor image in pytorch normalized at 0.5
            1x3xwxh
    belief: tensor of the same size as the image to overlay over img
            1xnb_beliefxwxh
    name: str to name the image, e.g., output.png
    path: where to save, e.g., /where/to/save/
    factor: float [0,1] how much to keep the original, 1 = fully, 0 black
    grid: how big the grid, e.g., 3 wide.
    norm_belief: bool to normalize the values [0,1]
    """

    tensor = torch.squeeze(beliefs)
    belief_imgs = []
    in_img = torch.squeeze(img)
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize([in_img.size()[1], in_img.size()[2]]), transforms.ToTensor()])
    in_img *= factor
    norm_belief = True
    for j in range(tensor.size()[0]):
        belief = tensor[j].clone()
        if norm_belief:
            belief -= float(torch.min(belief).data.cpu().numpy())
            belief /= float(torch.max(belief).data.cpu().numpy())
        belief = torch.clamp(belief, 0, 1).cpu()
        belief = torch.squeeze(transform(belief.unsqueeze(0))).cuda()
        belief = torch.cat([
            belief.unsqueeze(0) + in_img[0, :, :],
            belief.unsqueeze(0) + in_img[1, :, :],
            belief.unsqueeze(0) + in_img[2, :, :]
        ]).unsqueeze(0)
        belief = torch.clamp(belief, 0, 1).cpu()

        belief_imgs.append(belief.data.squeeze().numpy())

    # Create the image grid
    belief_imgs = torch.tensor(np.array(belief_imgs))

    save_image(belief_imgs, "{}{}".format(path, name),
               mean=0, std=1, nrow=grid)

irange = range

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize == True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each == True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=4, padding=2, mean=None, std=None):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image

    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=10, pad_value=1)
    if not mean is None:
        ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    else:
        ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def run_dope_node(params, freq=5, overlaybelief=False):
    '''Starts ROS node to listen to image topic, run DOPE, and publish DOPE results'''

    global g_img
    global g_draw

    pubs = {}
    models = {}
    pnp_solvers = {}
    pub_dimension = {}
    draw_colors = {}

    # Initialize parameters
    matrix_camera = np.zeros((3,3))
    matrix_camera[0,0] = params["camera_settings"]['fx']
    matrix_camera[1,1] = params["camera_settings"]['fy']
    matrix_camera[0,2] = params["camera_settings"]['cx']
    matrix_camera[1,2] = params["camera_settings"]['cy']
    matrix_camera[2,2] = 1
    dist_coeffs = np.zeros((4,1))

    if "dist_coeffs" in params["camera_settings"]:
        dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = params['thresh_angle']
    config_detect.thresh_map = params['thresh_map']
    config_detect.sigma = params['sigma']
    config_detect.thresh_points = params["thresh_points"]


    # search all the light field images
    print(matrix_camera)
    img_path = []
    for root, dirnames, filenames in os.walk(params['image_folder']):
        for filename in fnmatch.filter(filenames,params['image_format']):
            img_path.append(os.path.join(root, filename))
    # print(img_path)

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model in params['weights']:
        models[model] =\
            ModelData(
                model, 
                g_path2package + "/weights/" + params['weights'][model]
            )
        models[model].load_net_model()
        
        draw_colors[model] = \
            tuple(params["draw_colors"][model])
        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                matrix_camera,
                Cuboid3d(params['dimensions'][model]),
                dist_coeffs=dist_coeffs
            )
        # pubs[model] = \
        #     rospy.Publisher(
        #         '{}/pose_{}'.format(params['topic_publishing'], model),
        #         PoseStamped,
        #         queue_size=10
        #     )
        # pub_dimension[model] = \
        #     rospy.Publisher(
        #         '{}/dimension_{}'.format(params['topic_publishing'], model),
        #         String,
        #         queue_size=10
        #     )

    # Start ROS publisher
    # pub_rgb_dope_points = \
    #     rospy.Publisher(
    #         params['topic_publishing']+"/rgb_points", 
    #         ImageSensor_msg, 
    #         queue_size=10
    #     )
    
    # Starts ROS listener
    # rospy.Subscriber(
    #     topic_cam, 
    #     ImageSensor_msg, 
    #     __image_callback
    # )

    # Initialize ROS node
    # rospy.init_node('dope_save_to_file', anonymous=True)
    # rate = rospy.Rate(freq)

    print ("Running DOPE...  (Processing the folder: '{}')".format(params['image_folder'])) 
    print ("Ctrl-C to stop")

    # while not rospy.is_shutdown():
        # if g_img is not None:
    for sub_ap_img in img_path:
        print(sub_ap_img)
        # Copy and draw image

        # g_img = cv2.imread(sub_ap_img)
        # g_img = cv2.cvtColor(g_img, cv2.COLOR_RGB2BGR)

        g_img = cv2.imread(sub_ap_img,cv2.IMREAD_GRAYSCALE)
        g_img = cv2.cvtColor(g_img,cv2.COLOR_GRAY2BGR)
        img_copy = g_img.copy()
        im = Image.fromarray(img_copy)
        g_draw = ImageDraw.Draw(im)

        for m in models:
            # Detect object
            if overlaybelief:
                belief_tensor, img_tensor = ObjectDetector.retrive_belief_map(models[m].net, g_img)
                OverlayBeliefOnImage(img_tensor, belief_tensor, name=(sub_ap_img[:-4] + 'beliefmap.png'), factor=0.5)



            else:
                results = ObjectDetector.detect_object_in_image(
                    models[m].net,
                    pnp_solvers[m],
                    g_img,
                    config_detect
                )
                # Publish pose and overlay cube on image
                for i_r, result in enumerate(results):
                    if result["location"] is None:
                        continue
                    print("have results!" + sub_ap_img)

                    # Draw the cube
                    if None not in result['projected_points']:
                        points2d = []
                        for pair in result['projected_points']:
                            points2d.append(tuple(pair))
                        DrawCube(points2d, draw_colors[m])
            # cv2.imwrite(sub_ap_img[:-4] + 'result.png', cv2.cvtColor(np.array(im),cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    '''Main routine to run DOPE'''

    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "config_pose.yaml"
    # rospack = rospkg.RosPack()
    params = None
    yaml_path = g_path2package + '/config/{}'.format(config_name)
    with open(yaml_path, 'r') as stream:
        try:
            print("Loading DOPE parameters from '{}'...".format(yaml_path))
            params = yaml.load(stream)
            print('    Parameters loaded.')
        except yaml.YAMLError as exc:
            print(exc)

    topic_cam = params['topic_camera']

 
    run_dope_node(params,overlaybelief=params['overlaybelief'])

