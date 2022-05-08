import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

source = "data/trevor_rocket_1/"
frame_list = os.listdir(source)
frame_list.sort(key=lambda x: (len(x), x))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('diff_trevor.mp4',fourcc, 30, (1080,1920))


bbox_loose = [1050, 500, 1200, 550]
bbox_tight_list = [[]]

def drawbbox(img, bbox):
    img[bbox[0], bbox[1]:bbox[3]] = 255
    img[bbox[2], bbox[1]:bbox[3]] = 255
    img[bbox[0]:bbox[2], bbox[1]] = 255
    img[bbox[0]:bbox[2], bbox[3]] = 255
    return img

def maskbbox(shape, bbox):
    mask = np.zeros(shape)
    mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
    return mask

prev = cv2.imread(source + frame_list[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)

for i in range(len(frame_list)-1):
    next = cv2.imread(source + frame_list[i + 1], cv2.IMREAD_GRAYSCALE).astype(np.float32)

    farneback_params = dict(pyr_scale=0.8, levels=15, winsize=5,
                                      iterations=10, poly_n=5, poly_sigma=0,
                                      flags=10)

    #mask = maskbbox(next.shape, bbox_loose)
    #flow = cv2.calcOpticalFlowFarneback(prev=prev, next=next, flow=None, **farneback_params)
    #magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    #mask = np.zeros((prev.shape[0], prev.shape[1], 3))
    #mask[:, :, 1] = 100
    #mask[:, :, 0] = angle * 180 / np.pi / 2
    #mask[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    #mask = mask.astype(np.uint8)
    #motion = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR).astype(np.uint8)

    motion = np.log(abs((next-prev))) * 10
    motion = motion
    motion = cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR)

    motion[motion > 255] = 255
    motion[motion < 0] = 0

    motion = motion.astype(np.uint8)
    """
    Display video
    """
    #img = drawbbox(prev, bbox_loose).astype(np.uint8)
    print(i)


    out.write(motion)

    prev = next
out.release()