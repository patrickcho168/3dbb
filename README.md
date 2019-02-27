# 3D Bounding Box Estimation Using Deep Learning and Geometry
If interested, join the slack workspace where the paper is discussed, issues are worked through, and more! Click this [link](https://join.slack.com/t/3dboundingbox-oun9186/shared_invite/enQtNDk4Njg2NzYyNzY5LWVlZWRlMjNhZmZlYjVmNGY3NWVlNDA4MmY2ZWQ3ZmUyY2Q4OWIxMmY4NzU4YmViM2ViZWI5YjgxOTIyOTI4ZjI) to join.

## Introduction
PyTorch implementation for this [paper](https://arxiv.org/abs/1612.00496).

![example](http://soroushkhadem.com/img/2d-top-3d-bottom1.png)

## How it works

## Usage
First, download the data from [Kitti](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
You will need the left color images, the training labels, and the camera calibration matrices. Total is ~13GB.


## Training


## Future Goals
- YOLO integration to get 2D boxes
- Cuda optimization to run frame by frame on video feed
- ROS node to publish positions

## Credit
I originally started from a fork of this [repo](https://github.com/fuenwang/3D-BoundingBox), and some of the original code still exists in the training script.
