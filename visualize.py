import os
import pandas as pd
import numpy as np
import cv2

def draw(src, points):

    if src[-3:] != 'jpg':
        src = src + '.jpg'
    path = os.path.join('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\RektNet_Dataset',src)
    # print(path)

    img = cv2.imread(path)
    colors = [[255, 255, 255], [147, 20, 255], [255, 0, 0], [0, 0, 0], [0, 100, 0], [211,0,148], [0, 0, 255]]
    # [white, pink, blue, black, green, purple, red] 

    if img is not None:
        res=img.copy()
        sz = (min(img.shape[0], img.shape[1])//30)
        if sz ==   0: sz = 1
        for idx,point in enumerate(points):
            coords = pd.eval(point)
            cv2.circle(res,coords,sz,colors[idx],-1)

        res = cv2.resize(res, (160,160     ))
        # cv2.imshow(src, res)
        cv2.imshow('dsarasfdsafdsafdsadfsadfsadfsa', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

names = os.listdir('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\RektNet_Dataset') #path to directory containg images
print(len(names))
       
# labels = pd.read_csv('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\rektnet_label.csv') #path to labels csv
# labels = pd.read_excel('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\dataset\\RektNet Data JDEs.xlsx') #path to labels csv
# labels = pd.read_excel('output_test.xlsx')
labels = pd.read_excel('D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\outputs\\december-2021-experiments\\trial_new_dataset             \\model_predictions.xlsx')

fnames = list(labels.values[:, 0])
points = labels.values[:, 2:9]
# print(points)
# print(points.shape[0])

test = fnames#    [10:14] #select image indices to visualize
# print(test)

for fname in test:
    fname_idx = fnames.index(fname)
    fpoints = points[fname_idx]
    # print(fname)
    # print(fpoints)
    draw(fname, fpoints)

"""
for i in range(len(labels)):
    for j in range(7):
        ls_ = labels.iloc[i, j + 2]
        ls = [int(ls_.split(',')[0][1:]), int(ls_.split(',')[1][:-1])]
        # print(ls)
        if ls[0] == 0 or ls[1] == 0:
            print(f'zero index occurs at row id {i} column id {j+2} column name {labels.columns[j+2]}')
"""