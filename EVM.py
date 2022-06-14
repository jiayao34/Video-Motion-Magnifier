#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:57:08 2022

@author: jiayaoyuan
"""

import cv2
import numpy as np
import scipy.signal as signal

#load video
def load_video(video_filename):
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
    x = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            video_tensor[x] = frame
            x += 1
        else:
            break
    return video_tensor, fps

#Build Laplacian Pyramid
def build_laplacian_pyramid(src,levels=3):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    pyramid=[]
    for i in range(levels,0,-1):
        GE = cv2.pyrUp(gaussianPyramid[i])
        L = cv2.subtract(gaussianPyramid[i-1], GE)
        pyramid.append(L)
    return pyramid

#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=3):
    s = src.copy()
    pyramid = [s]
    for i in range(level):
        s = cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid

#build laplacian pyramid for video
def laplacian_video(video_tensor, levels=3):
    tensor_list=[]
    for i in range(0, video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_laplacian_pyramid(frame,levels=levels)
        if i == 0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0], pyr[k].shape[0], pyr[k].shape[1], 3)))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
    return tensor_list

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

def reconstract_from_tensorlist(filter_tensor_list,levels=3):
    final = np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up = cv2.pyrUp(up) + filter_tensor_list[n + 1][i]
            # up = cv2.pyrUp(up)
        final[i] = up
    return final

def save_video(video_tensor):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("amplifiedVideo.avi", fourcc, 30, (width, height), 1)
    for i in range(0, video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()
    
low, high = 0.4, 3
levels = 3
amplification = 20
t, f = load_video('baby.mp4')
lap_video_list = laplacian_video(t, levels=levels)
filter_tensor_list=[]
for i in range(levels):
    filter_tensor = butter_bandpass_filter(lap_video_list[i],low,high,f)
    filter_tensor *= amplification
    filter_tensor_list.append(filter_tensor)
recon = reconstract_from_tensorlist(filter_tensor_list)
final = t + recon
save_video(final)