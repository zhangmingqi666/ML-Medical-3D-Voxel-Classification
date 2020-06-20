import pandas as pd     
import numpy as np
import keras
import os
import os.path
import sys
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,MaxPooling2D,BatchNormalization,Activation
from keras.optimizers import Adam
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from mylib.models import densesharp, metrics, losses,DenseNet
from mylib.models.DenseNet import createDenseNet
from keras.optimizers import SGD
import pandas as pd
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

x_path='/content/drive/My Drive/MachineLeaning/train_val/train_val'
x_file=os.listdir(x_path)
x_filef_train_set=x_file[0:464]
x_test_path='/content/drive/My Drive/MachineLeaning/test/test'
size=32


def get_dataset():
    x_return_train = np.zeros((465, size, size, size))
    x_name=pd.read_csv("train_val.csv") ['name']
    filenum = 0
    for i in range(len(x_filef_train_set)):
        x_file_temp=os.path.join(x_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask*0.8+x_voxel*0.2

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
        # print(file, coor1low, coor1high, coor2low, coor2high, coor3low, coor3high)
        coorlist1 = 0
        for coor1 in range(coor1low, coor1high):
            coorlist2 = 0
            for coor2 in range(coor2low, coor2high):
                coorlist3 = 0
                for coor3 in range(coor3low, coor3high):
                    # xx[filenum, coor1, coor2, coor3] = list_xx[coor1min+coorlist1, coor2min+coorlist2, coor3min+coorlist3]
                    x_return_train[filenum, coor1, coor2, coor3] = x_temp[
                        coor1min + coorlist1, coor2min + coorlist2, coor3min + coorlist3]
                    coorlist3 += 1
                coorlist2 += 1
            coorlist1 += 1
        filenum += 1
    return  x_return_train


def get_label():
    x_label=pd.read_csv("train_val.csv") ['label']
    x_tr_label=keras.utils.to_categorical(x_label,2)[0:465]
    return x_tr_label


def get_testdataset():
    x_return = np.zeros((117, size, size, size))
    x_name=pd.read_csv("test_2.csv") ['name']
    filenum=0
    for i in range(117):
        x_file_temp=os.path.join(x_test_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask*0.8+x_voxel*0.2
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

def get_batch(x, y, step, batch_size, alpha=0.2):
    candidates_data, candidates_label = x, y
    offset = (step * batch_size) % (candidates_data.shape[0] - batch_size)

    # get batch data
    train_features_batch = candidates_data[offset:(offset + batch_size)]
    train_labels_batch = candidates_label[offset:(offset + batch_size)]

    # 最原始的训练方式
    if alpha == 0:
        return train_features_batch, train_labels_batch
    # mixup增强后的训练方式
    if alpha > 0:
        weight = np.random.beta(alpha, alpha, batch_size)
        x_weight = np.zeros((batch_size, 32,32,32,1))
        for jj in range(batch_size):
            for hh in range(32):
                for gg in range(32):
                    for tt in range(32):
                        x_weight[jj,hh,gg,tt,0] = weight[jj]
        y_weight = weight.reshape(batch_size, 1)
        index = np.random.permutation(batch_size)
        y1, y2 = train_labels_batch, train_labels_batch[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        x1, x2 = train_features_batch, train_features_batch[index]
        x = x1* x_weight + x2* (1 - x_weight)
        return x, y

def get_batch(x1, y1, alpha):
    """Mix data
    x1: input numpy array.
    y1: target numpy array.
    alpha: float.
    Paper url: https://arxiv.org/pdf/1710.09412.pdf
    See Figure1.
    Return
        mixed_x, y_a, y_b, lam
    """
    x2=np.zeros(np.shape(x1))
    y2=np.zeros(np.shape(y1),'float')
    n = len(x1)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, n)
    else:
        lam = np.array([1.0] * n)
    indexs = np.random.randint(0, n, n)
    for i in range(n):
        x2[i] = x1[i]*lam[i]+(1-lam[i])*x1[indexs[i]]
        y2[i] = y1[i]*lam[i]+(1-lam[i])*y1[indexs[i]]
    return x2, y2

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

densenet_depth = 10 # Must be 3N+4
densenet_growth_rate = 12
batch_size = 512
x_train_set=get_dataset()
x_train_set=np.array(x_train_set)
x_train_label=get_label()
x_train_set=x_train_set.reshape(x_train_set.shape[0], 32, 32, 32, 1)
x_train_set=x_train_set.astype('float32')/255
"""
x_train,x_train_label=get_batch(x_train,x_train_label,1,8,0.2)
x_train, x_train_label=get_batch(x_train_set, x_train_label, 0.2)
"""
x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train_set, x_train_label, test_size=0.2, random_state=3)

x_predict = np.array(get_testdataset())
x_predict = x_predict.reshape(x_predict.shape[0], 32, 32, 32, 1)
x_predict = x_predict.astype('float32')/255

early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
nb_classes = 2
saveBestModel = keras.callbacks.ModelCheckpoint('./bestweight_xdczytnl.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model = createDenseNet(nb_classes=nb_classes, img_dim=[32,32,32,1], depth=densenet_depth, growth_rate=densenet_growth_rate)
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.summary()  # print the model


model.fit(x_train_train, y_train_train,batch_size= batch_size, epochs=2000, validation_data=(x_train_test, y_train_test), verbose=2, shuffle=False, callbacks=[early_stopping,saveBestModel])
loss,accuracy = model.evaluate(x_train_train,y_train_train)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss,accuracy))
loss,accuracy = model.evaluate(x_train_test,y_train_test)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (loss,accuracy))

print(model.predict(x_predict))
model.save("Alexstrasza.h5")

model = load_model('Alexstrasza.h5')
'''
model=load_model('model_xbt.h5')
model.load_weights('bestweight.h5')
'''
y = model.predict(x_predict)
y_pred=pd.DataFrame(y)
y_pred.to_csv('Alexstrasza.csv')

print('--------END---------')