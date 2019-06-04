"""
This script will use the 2D box from the label rather than from YOLO
"""
from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages

import os
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
import numpy as np

# to run car by car
single_car = False

def plot_regressed_3d_bbox(img, truth_img, cam_to_img, box_2d, dimensions, alpha, theta_ray):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box_2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print ('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        #TODO model in Cuda throws an error
        model = Model.Model(features=my_vgg.features, bins=2)
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/../nusc_kitti/mini_val')
    averages = ClassAverages.ClassAverages()
    all_images = dataset.all_objects()
    orient_score = 0
    l2 = 0
    tot = 0
    os_tot = 0
    for key in sorted(all_images.keys()):
        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']

        for object in objects:
            label = object.label
            theta_ray = object.theta_ray
            input_img = object.img

            input_tensor = torch.zeros([1,3,224,224])
            input_tensor[0,:,:,:] = input_img
            input_tensor.cuda()

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(label['Class'])

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += dataset.angle_bins[argmax]
            alpha -= np.pi
            delta_theta = label['Alpha'] - alpha
            tot += 1
            if label['Class'] != 'traffic_cone':
                orient_score += (1 + np.cos(delta_theta))/2
                os_tot += 1
            label_dim = label['Dimensions']
            l2 += (dim[0]-label_dim[0])**2 + (dim[1]-label_dim[1])**2 + (dim[2]-label_dim[2])**2
            print('Average Orientation Score', orient_score/os_tot)
            print('L2 Loss', l2/tot)
            print('Total Orientation Examples', os_tot)
            print('Total Examples', tot)
            location = plot_regressed_3d_bbox(img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)

            print('Estimated pose: %s'%location)
            print('Truth pose: %s'%label['Location'])
            print('-------------')

            # plot car by car
            if single_car:
                numpy_vertical = np.concatenate((truth_img, img), axis=0)
                #cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
                #cv2.waitKey(0)
                cv2.imwrite(os.path.join('output', key + '_yolo.png'), numpy_vertical)

        # plot image by image
        if not single_car:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imwrite(os.path.join('output', key + '_yolo.png'), numpy_vertical)
            #cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
            #cv2.waitKey(0)

if __name__ == '__main__':
    main()
