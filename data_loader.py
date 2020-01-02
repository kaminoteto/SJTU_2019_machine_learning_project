#import cv2
import csv
import os

import pandas as pd


import numpy as np


#

#files_path = "./train_val"


def file5array(file_path, batch_size,variaty):


    #filenames = os.listdir(file_path)

    #file_num = len(filenames)
    #print(image_num)



    database = pd.read_csv('train_val.csv')
    A_lable = database['lable']
    B_name = database['name']
    A=np.array(A_lable)
    B=np.array(B_name)
   # file_num = B.shape

    File = np.load('./train_val/' + B[0]+'.npz')
    File = File['voxel']
    File = File[34:66, 34:66, 34:66]
    File = File.reshape(1, File.shape[0], File.shape[1], File.shape[2], 1)
    File_sum = File

    idx=0
    for k in range(1, 465):
        File = np.load('./train_val/'+B[k]+'.npz')
        File=File['voxel']
        File = File[34:66, 34:66, 34:66]
        File = File.reshape(1, File.shape[0], File.shape[1], File.shape[2],1)
        File_sum = np.concatenate((File_sum, File), axis=0)

    data = File_sum


    if variaty =='train':
        data = data [:360]
        name = B [:360]
        A = A[:360]
        num=360
    elif variaty == 'test':
        data = data [360:]
        name = B [360:]
        A = A[360:]
        num=105
    else: print('file5array wrong')
    while(1):
        if idx == 0:
            index = [i for i in range(len(A))]
            np.random.shuffle(index)
            data = data[index]
            A = A[index]
        x_cube = data[idx * batch_size:(idx + 1) * batch_size]
        y_cube = A[idx * batch_size:(idx + 1) * batch_size]
        name_cube   = name[idx * batch_size:(idx + 1) * batch_size]
        idx += 1
        '''
        index = [i for i in range(len(A))]
        np.random.shuffle(index)
        data = data[index]
        A = A[index]
        '''
        #idx = 0 if idx > (num // batch_size)-1 else idx
        if idx > (num // batch_size) - 1:
            idx = 0
            #seed = 100
            #random.seed(seed)
            #random.shuffle(data)
            #random.seed(seed)  #
            #random.shuffle(A)
            index = [i for i in range(len(A))]
            np.random.shuffle(index)
            data = data[index]
            A = A[index]

        yield x_cube, y_cube

def file5ori_array(file_path):
    database = pd.read_csv('sampleSubmission.csv')
    #A_lable = database['lable']
    B_name = database['name']
   # A = np.array(A_lable)
    B = np.array(B_name)
    # file_num = B.shape

    File = np.load('./test/' + B[0] + '.npz')
    File = File['voxel']
    File = File[34:66, 34:66, 34:66]
    File = File.reshape(1, File.shape[0], File.shape[1], File.shape[2], 1)
    File_sum = File

    idx = 0
    for k in range(1, 117):
        File = np.load('./test/' + B[k] + '.npz')
        File = File['voxel']
        File = File[34:66, 34:66, 34:66]
        File = File.reshape(1, File.shape[0], File.shape[1], File.shape[2], 1)
        File_sum = np.concatenate((File_sum, File), axis=0)

    data = File_sum
    return data
def file5ori_array_1(file_path):
    database = pd.read_csv('train_val.csv')
    A_lable = database['lable']
    B_name = database['name']
    A = np.array(A_lable)
    B = np.array(B_name)
    # file_num = B.shape

    File = np.load('./train_val/' + B[0] + '.npz')
    File = File['voxel']
    File = File[34:66, 34:66, 34:66]
    File = File.reshape(1, File.shape[0], File.shape[1], File.shape[2], 1)
    File_sum = File

    idx = 0
    for k in range(1, 465):
        File = np.load('./train_val/' + B[k] + '.npz')
        File = File['voxel']
        File = File[34:66, 34:66, 34:66]
        File = File.reshape(1, File.shape[0], File.shape[1], File.shape[2], 1)
        File_sum = np.concatenate((File_sum, File), axis=0)

    data = File_sum
    return data


#if __name__  == "__main__":

    #cube = file5array(files_path)
    #print(cube.shape)
    #print(filearray())
#print(cube[0,:,:,:,0])
#for i in range(0,32,2):
    #plt.figure()
    #plt.imshow(cube[0,i,:,:,0])
    #plt.show()

