import cv2
import h5py
import numpy as np                  # 实际使用时，需要先在指定文件夹中预先存放一张图片
import matplotlib.pylab as plt
from scipy.misc import imsave       #python scipy.misc.imsave()函数此功能仅在安装了Python Imaging Library（PIL）时可用。

from skimage import transform
 
 
def load_dataset():
 
    train_dataset = h5py.File('datasets/yolo.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
 
    test_dataset = h5py.File('datasets/yolo.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
 
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
 
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
 
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
 
 
def processing():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
 
 
    m = len(X_train_orig)
    # print(X_train_orig[1].shape)      # 输出一个例子
 
    Y_train_t = Y_train_orig.T
 
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(X_train_orig[i])
        plt.title(Y_train_t[i])
        plt.axis('off')

    plt.show()
 
    for i in range(m):
        name = 'images/train/' + str(i) + '.jpg'            # 在此文件夹下，先弄个“0.jpg”图片才行
        imsave(name, transform.rescale(X_train_orig[i].reshape(64, 64, 3), 1, mode='constant'))


if __name__ == '__main__':
    processing()