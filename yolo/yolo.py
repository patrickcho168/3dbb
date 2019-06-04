"""
Will use opencv's built in darknet api to do 2D object detection which will
then get passed into the torch net

source: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
"""

import cv2
import numpy as np
import os
from models import *
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.datasets import pad_to_square, resize
from utils.utils import *

class cv_Yolo:

    def __init__(self, yolo_path, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold

        # labels_path = os.path.sep.join([yolo_path, "coco.names"])
        labels_path = os.path.sep.join([yolo_path, "classes.names"])
        self.labels = open(labels_path).read().split("\n")

        np.random.seed(42)
        self.colors = np.random.randint(0,255, size=(len(self.labels), 3), dtype="uint8")

        #weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
        #cfg_path = os.path.sep.join([yolo_path, "yolov3.cfg"])
        weights_path = os.path.sep.join([yolo_path, "exp1_ckpt_8.pth"])
        cfg_path = os.path.sep.join([yolo_path, "yolov3-custom.cfg"])
        self.net = Darknet(cfg_path).to('cuda') # OR cpu
        self.net.load_state_dict(torch.load(weights_path))
        #self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    def detect(self, img_path):
        # assert image is opencv
        # (H,W) = image.shape[:2]

        #ln = self.net.getLayerNames()
        #ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # prepare input
        # blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # blob = torch.Tensor(blob).to('cuda')

        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        _, H, W = img.shape
        h_factor, w_factor = (H, W)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        imgs = torch.stack([resize(img_, 416) for img_ in [img]]).to('cuda')

        #self.net.setInput(blob)
        #output = self.net.forward(ln)
        with torch.no_grad():
            output = self.net(imgs)
            output = non_max_suppression(output, 0.5, 0.5)[0]
        # output = self.net.forward(imgs).numpy()
        output = rescale_boxes(output, 416, (H, W)).numpy()
        detections = []

        boxes = []
        confidences = []
        class_ids = []

        # for output in output:
        #for detection in output:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in output:
                #scores = detection[5:]
                #class_id = np.argmax(scores)
                #confidence = 1.0 # min(detection[4], scores[class_id])

                #if confidence > self.confidence:

                    #box = detection[0:4] / np.array([416, 416, 416, 416]) * np.array([W, H, W, H])
                    #(centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    width = x2 - x1
                    height = y2 - y1
                    x = x1#int(centerX - (width / 2))
                    y = y1#int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(cls_conf))
                    class_ids.append(int(cls_pred))
                    class_ = self.get_class(int(cls_pred))
                    top_left = (int(x1), int(y1))
                    bottom_right = (int(x2), int(y2))
                    box_2d = [top_left, bottom_right]
                    detections.append(Detection(box_2d, class_))



        #idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        #if len(idxs) > 0:
        #    for i in idxs.flatten():

        #        top_left = (boxes[i][0], boxes[i][1])
        #        bottom_right = (top_left[0] + boxes[i][2], top_left[1] + boxes[i][3])

        #        box_2d = [top_left, bottom_right]
        #        class_ = self.get_class(class_ids[i])
        #        if class_ == "person":
        #            class_ = "pedestrian"

        #        detections.append(Detection(box_2d, class_))

        return detections

    def get_class(self, class_id):
        return self.labels[class_id]



class Detection:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_
