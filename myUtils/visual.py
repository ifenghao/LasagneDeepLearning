# coding:utf-8
import numpy as np
from matplotlib import pylab
import os

img_count = 0
username = 'zhufenghao'


def addPad(map2D, padWidth):
    row, col = map2D.shape
    hPad = np.zeros((row, padWidth))
    map2D = np.hstack((hPad, map2D, hPad))
    vPad = np.zeros((padWidth, col + 2 * padWidth))
    map2D = np.vstack((vPad, map2D, vPad))
    return map2D


def squareStack(map3D):
    mapNum = map3D.shape[0]
    row, col = map3D.shape[1:]
    side = int(np.ceil(np.sqrt(mapNum)))
    lack = side ** 2 - mapNum
    map3D = np.vstack((map3D, np.zeros((lack, row, col))))
    map2Ds = [addPad(map3D[i], 1) for i in range(side ** 2)]
    return np.vstack([np.hstack(map2Ds[i:i + side])
                      for i in range(0, side ** 2, side)])


def show_beta(beta):
    size, channels = beta.shape
    beta = beta.T.reshape((channels, int(np.sqrt(size)), int(np.sqrt(size))))
    beta = squareStack(beta)
    pylab.figure()
    pylab.gray()
    pylab.imshow(beta)
    pylab.show()


def save_beta(beta, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    size, channels = beta.shape
    beta = beta.T.reshape((channels, int(np.sqrt(size)), int(np.sqrt(size))))
    pylab.figure()
    pylab.gray()
    pylab.imshow(squareStack(beta))
    pylab.savefig(pic_path)
    pylab.close()


def save_beta_mch(beta, mch, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    size, channels = beta.shape
    size /= mch
    beta = np.split(beta.T, mch, axis=1)
    beta = np.concatenate(beta, axis=0)
    beta = beta.reshape((mch * channels, int(np.sqrt(size)), int(np.sqrt(size))))
    pylab.figure()
    pylab.gray()
    pylab.imshow(squareStack(beta))
    pylab.savefig(pic_path)
    pylab.close()


def show_map(map):
    num = len(map)
    pylab.figure()
    pylab.gray()
    for i in xrange(num):
        pylab.subplot(1, num, i + 1)
        pylab.imshow(squareStack(map[i]))
    pylab.show()


def save_map(map, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    num = len(map)
    pylab.figure()
    pylab.gray()
    for i in xrange(num):
        pylab.subplot(1, num, i + 1)
        pylab.imshow(squareStack(map[i]))
    pylab.savefig(pic_path)
    pylab.close()
