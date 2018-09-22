import gzip
import os
import numpy as np
import cv2
import sklearn.datasets
import random
import sys
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

py_version = sys.version[0]

def load_mnist(data_dir):

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8)
        return data

    data = extract_data(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), 60000, 16, 28*28)
    trainX = data.reshape((60000, 28, 28, 1))

    data = extract_data(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'), 60000, 8, 1)
    trainY = data.reshape((60000)).astype(np.int32)

    data = extract_data(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'), 10000, 16, 28*28)
    testX = data.reshape((10000, 28, 28, 1))

    data = extract_data(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'), 10000, 8, 1)
    testY = data.reshape((10000)).astype(np.int32)

    return trainX, trainY, testX, testY


def load_svhn(data_dir):
    train_data = sio.loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    trainX = train_data['X']
    trainX = np.transpose(trainX, [3,0,1,2])
    trainY = train_data['y']
    return trainX, trainY


def load_cifar10(data_dir):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            if py_version == '3':
                dt = pickle.load(fo, encoding='bytes')
            else:
                dt = pickle.load(fo)                
        data = np.array(dt[b'data'], dtype=np.uint8).reshape([-1,3,32,32])
        return np.transpose(data, axes=[0, 2, 3, 1]), \
               np.array(dt[b'labels'], dtype=np.int32)
    
    filename_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    trainX_list = []
    trainY_list = []
    for filename in filename_list:
        path = os.path.join(data_dir, 'cifar-10-batches-py', filename)
        data, labels = unpickle(path)
        trainX_list.append(data)
        trainY_list.append(labels)
    trainX = np.concatenate(trainX_list, axis=0)
    trainY = np.concatenate(trainY_list, axis=0)

    path = os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch')
    testX, testY = unpickle(path)
    return trainX, trainY, testX, testY


def get_epoch(X, Y, batch_size):
    ang = np.arange(X.shape[0])
    np.random.shuffle(ang)
    X = X[ang]
    Y = Y[ang]
    for i in range(X.shape[0]//batch_size):
        yield (X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size])


def get_batch(X, Y, batch_size):
    while True:
        for image, label in get_epoch(X, Y, batch_size):
            yield (image, label)


def inverse_transform(images, norm='[-1,1]'):
    if norm == '[-1,1]':
        return ((images+1.)*(255./2.)).astype(np.uint8)
    else:
        return (images*255.).astype(np.uint8)


def merge(images, size):
    assert size[0]*size[1] <= images.shape[0], 'The number of images is not enough!'
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h*size[0], w*size[1], c))
        for idx, image in enumerate(images[:size[0]*size[1]]):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:(j+1)*h, i*w:(i+1)*w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h*size[0], w*size[1]))
        for idx, image in enumerate(images[:size[0]*size[1]]):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:(j+1)*h, i*w:(i+1)*w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


def save_images(images, size, image_path, norm='[-1,1]'):
    images = merge(inverse_transform(images, norm), size)
    cv2.imwrite(image_path, images)


def save_plot(x, y, image_path, title_name, y_label_name, x_label_name='iteration'):
    plt.clf()
    plt.plot(x, y)
    plt.ylabel(y_label_name)
    plt.xlabel(x_label_name)
    plt.title(title_name)
    plt.savefig(image_path) 

