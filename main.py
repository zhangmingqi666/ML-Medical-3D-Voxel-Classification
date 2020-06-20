import pandas as pd
from matplotlib import numpy as np
import keras
import os
import os.path
from mylib.models.DenseNet import createDenseNet
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model
import math
##路径
traindata_path = './train_val/train_val'
file_list = os.listdir(traindata_path)
traindata_list = file_list[0:495]
x_test_path = './test/test'
size = 32

os.environ['KERAS_BACKEND'] = 'tensorflow'


def load_traindata():
    x_return = np.zeros((495, size, size, size))
    x_name = pd.read_csv("train_val.csv")['name']   
    count_file = 0
    for i in range(len(traindata_list)):
        x_file_temp = os.path.join(traindata_path, x_name[i]+'.npz')
        x_voxel = np.array(np.load(x_file_temp)['voxel'])
        x_mask = np.array(np.load(x_file_temp)['seg'])
        x_temp = x_voxel*x_mask*0.8+x_voxel*0.2

        list_xx = x_mask * x_voxel
        list_xx_nz = np.array(np.nonzero(list_xx))
        coor1min = list_xx_nz[0, :].min()
        coor1max = list_xx_nz[0, :].max()
        coor1len = coor1max - coor1min + 1
        coor1bigger = coor1len - size
        if coor1bigger > 0:
            coor1min += coor1bigger // 2
            coor1max -= coor1bigger - coor1bigger // 2
            coor1len = size
        coor1low = (size // 2) - (coor1len // 2)
        coor1high = coor1low + coor1len
        coor2min = list_xx_nz[1, :].min()
        coor2max = list_xx_nz[1, :].max()
        coor2len = coor2max - coor2min + 1
        coor2bigger = coor2len - size
        if coor2bigger > 0:
            coor2min += coor2bigger // 2
            coor2max -= coor2bigger - coor2bigger // 2
            coor2len = size
        coor2low = (size // 2) - (coor2len // 2)
        coor2high = coor2low + coor2len
        coor3min = list_xx_nz[2, :].min()
        coor3max = list_xx_nz[2, :].max()
        coor3len = coor3max - coor3min + 1
        coor3bigger = coor3len - size
        if coor3bigger > 0:
            coor1min += coor3bigger // 2
            coor3max -= coor3bigger - coor3bigger // 2
            coor3len = size
        coor3low = (size // 2) - (coor3len // 2)
        coor3high = coor3low + coor3len
        coorlist1 = 0
        for coor1 in range(coor1low, coor1high):
            coorlist2 = 0
            for coor2 in range(coor2low, coor2high):
                coorlist3 = 0
                for coor3 in range(coor3low, coor3high):
                    x_return[count_file, coor1, coor2, coor3] = x_temp[
                        coor1min + coorlist1, coor2min + coorlist2, coor3min + coorlist3]
                    coorlist3 += 1
                coorlist2 += 1
            coorlist1 += 1
        count_file += 1
    return x_return


def load_testdata():
    x_return = np.zeros((117, size, size, size))
    x_name = pd.read_csv("sampleSubmission.csv")['name'] 
    filenum = 0
    for i in range(117):
        x_file_temp = os.path.join(x_test_path, x_name[i]+'.npz')
        x_voxel = np.array(np.load(x_file_temp)['voxel'])
        x_mask = np.array(np.load(x_file_temp)['seg'])
        x_temp = x_voxel*x_mask*0.8+x_voxel*0.2
        list_xx_test = x_voxel * x_mask
        list_xx_nz_test = np.array(np.nonzero(list_xx_test))
        coor1min = list_xx_nz_test[0, :].min()
        coor1max = list_xx_nz_test[0, :].max()
        coor1len = coor1max - coor1min + 1
        coor1bigger = coor1len - size
        if coor1bigger > 0:
            coor1min += coor1bigger // 2
            coor1max -= coor1bigger - coor1bigger // 2
            coor1len = size
        coor1low = (size // 2) - (coor1len // 2)
        coor1high = coor1low + coor1len
        coor2min = list_xx_nz_test[1, :].min()
        coor2max = list_xx_nz_test[1, :].max()
        coor2len = coor2max - coor2min + 1
        coor2bigger = coor2len - size
        if coor2bigger > 0:
            coor2min += coor2bigger // 2
            coor2max -= coor2bigger - coor2bigger // 2
            coor2len = size
        coor2low = (size // 2) - (coor2len // 2)
        coor2high = coor2low + coor2len
        coor3min = list_xx_nz_test[2, :].min()
        coor3max = list_xx_nz_test[2, :].max()
        coor3len = coor3max - coor3min + 1
        coor3bigger = coor3len - size
        if coor3bigger > 0:
            coor1min += coor3bigger // 2
            coor3max -= coor3bigger - coor3bigger // 2
            coor3len = size
        coor3low = (size // 2) - (coor3len // 2)
        coor3high = coor3low + coor3len
        # print(file, coor1low, coor1high, coor2low, coor2high, coor3low, coor3high)
        coorlist1 = 0
        for coor1 in range(coor1low, coor1high):
            coorlist2 = 0
            for coor2 in range(coor2low, coor2high):
                coorlist3 = 0
                for coor3 in range(coor3low, coor3high):
                    x_return[filenum, coor1, coor2, coor3] = x_temp[
                        coor1min + coorlist1, coor2min + coorlist2, coor3min + coorlist3]
                    coorlist3 += 1
                coorlist2 += 1
            coorlist1 += 1
        filenum += 1
    return x_return


def data_augmentation(x, y, index):
    x_lr = x_ud = x_r1 = x_r2 = x_r3 = x_r4 = x_r5 = x_r6 = x_m1 = x_m2 = x_m3 = x_m4 = np.zeros(np.shape(x))
    y_m1 = y_m2 = y_m3 = y_m4 = np.zeros(np.shape(y), 'float')

    l1 = np.random.beta(0.1, 0.1, len(x))
    l2 = np.random.beta(0.15, 0.15, len(x))
    l3 = np.random.beta(0.2, 0.2, len(x))
    l4 = np.random.beta(0.25, 0.25, len(x))
    randi = np.random.randint(0, len(x), len(x))
    for ii in range(x.shape[0]):
        x_lr[ii, :, :, :] = np.fliplr(x[ii, :, :, :])
        x_ud[ii, :, :, :] = np.flipud(x[ii, :, :, :])
        x_r1[ii, :, :, :] = np.rot90(x[ii, :, :, :], 1, (0, 1))
        x_r2[ii, :, :, :] = np.rot90(x[ii, :, :, :], 1, (0, 2))
        x_r3[ii, :, :, :] = np.rot90(x[ii, :, :, :], 1, (1, 2))
        x_r4[ii, :, :, :] = np.rot90(x[ii, :, :, :], 3, (0, 1))
        x_r5[ii, :, :, :] = np.rot90(x[ii, :, :, :], 3, (0, 2))
        x_r6[ii, :, :, :] = np.rot90(x[ii, :, :, :], 3, (1, 2))
        x_m1[ii] = x[ii] * l1[ii] + (1 - l1[ii]) * x[randi[ii]]
        y_m1[ii] = y[ii] * l1[ii] + (1 - l1[ii]) * y[randi[ii]]
        x_m2[ii] = x[ii] * l2[ii] + (1 - l2[ii]) * x[randi[ii]]
        y_m2[ii] = y[ii] * l2[ii] + (1 - l2[ii]) * y[randi[ii]]
        x_m3[ii] = x[ii] * l3[ii] + (1 - l3[ii]) * x[randi[ii]]
        y_m3[ii] = y[ii] * l3[ii] + (1 - l3[ii]) * y[randi[ii]]
        x_m4[ii] = x[ii] * l4[ii] + (1 - l4[ii]) * x[randi[ii]]
        y_m4[ii] = y[ii] * l4[ii] + (1 - l4[ii]) * y[randi[ii]]
    x_train = np.r_[x, x_lr, x_ud, x_r1, x_r2, x_m1, x_m2, x_m3, x_m4]
    y_train = np.r_[y, y, y, y, y, y_m1, y_m2, y_m3, y_m4]
    x_test = np.r_[x, x_lr, x_ud, x_r1, x_r2, x_r3, x_r4, x_r5, x_r6]
    y_test = np.r_[y, y, y, y, y, y, y, y, y]
    if index == 0:
        return x_train, y_train
    else:
        return x_test, y_test

set_predict = np.array(load_testdata())
set_predict = set_predict.reshape(set_predict.shape[0], 32, 32, 32, 1)
set_predict = set_predict.astype('float32')/255
print('End of data loading')

model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
print('End of model loading')

label_predict1 = model1.predict(set_predict)
label_predict2 = model2.predict(set_predict)
print('End of prediction')

output1 = pd.DataFrame(label_predict1)
output2 = pd.DataFrame(label_predict2)
output1.to_csv('Output1.csv')
output2.to_csv('Output2.csv')
score1 = pd.read_csv('Output1.csv')['1']
score2 = pd.read_csv('Output2.csv')['1']
score = 0.66 * score1 + 0.34 * score2
output = pd.read_csv('sampleSubmission.csv')
output['predicted'] = score
output.to_csv('Submission.csv', columns=['name', 'predicted'], index=False)   # 最终结果输出到Submission.csv中（若不存在将会被创建）
print('End of output')
