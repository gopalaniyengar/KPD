import numpy as np
import cv2
import os
import time
import pandas as pd
from tqdm import tqdm
import torch
import onnx
# from onnx2pytorch import ConvertModel

"""
GPU CUDA CUDNN
python train_eval.py --study_name trial_new_dataset

https://stackoverflow.com/questions/58833870/cant-we-run-an-onnx-model-imported-to-pytorch

TRAINING COMMAND: python train_eval.py --study_name new_dataset_train
OR nvm changed default training parameters in train_eval.py argparser
python train_eval.py --study_name geo_loss_train --checkpoint_interval 2 --geo_loss_gamma_vert 0.038 --geo_loss_gamma_horz 0.055 --evaluate_mode

INFERENCE COMMAND: python detect.py --model "D:\Python Projects\Rektnet\MITRepo\RektNet\outputs\december-2021-experiments\geo_loss_train\23_loss_0.38.pt" --img "D:\Python Projects\Rektnet\MITRepo\RektNet\dataset\RektNet_Dataset\vid_2_frame_1708_0.jpg"
"""

names = os.listdir('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\RektNet_Dataset') #path to directory containg images
labels = pd.read_csv('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\rektnet_label.csv') #path to labels csv
fnames = list(labels.values[:, 0])

names_excl = []
for name in names:
    if name not in fnames:
        names_excl.append(name)
print(f'Total: {len(names)}  =  Unlabelled: {len(names_excl)}  +  Labelled: {len(fnames)}')

def ms():
    return time.time_ns()/10**6

num_keypoints = 200
count = 0
tavg = 0
iters = int(len(names_excl)/num_keypoints) + 1
for i, name in enumerate(tqdm(names_excl)):
    if int(i%num_keypoints)==0:
        count += 1
        # print(f'saving keypoints....................................... {count}/{iters}')
        t1 = ms()
        # os.system(f'python .\\MITRepo\\RektNet\\detect.py --model "D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\outputs\\december-2021-experiments\\new_model\\best_keypoints_132x132.pt" --img "D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\RektNet_Dataset\\{name}" --study_name "new_model"')
        os.system(f'python .\\MITRepo\\RektNet\\detect.py --model "D:\Python Projects\\Rektnet\\MITRepo\\RektNet\\outputs\\december-2021-experiments\\hm_loss_try_160\\best_keypoints_160x160.pt" --img "D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\RektNet_Dataset\\{name}" --file --study_name "hm_loss_try_160"')
        t2 = ms()
        tavg += (t2-t1)/(iters)
print(f'Average execution time: {tavg}ms')