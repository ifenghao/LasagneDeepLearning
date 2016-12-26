# coding:utf-8
import numpy as np
import os, getpass
import cPickle
import scipy.io as sio
from scipy.misc import imread, imshow
from PIL import Image

username = 'zhufenghao'
datasets_dir = '/home/' + username + '/dataset'


def one_hot(x, n):
    x = np.array(x, dtype=np.int)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def listFiles(path, numPerDir=None):
    fileList = []
    try:
        dirs = os.listdir(path)
    except Exception:
        return []
    dirs = dirs[:numPerDir]
    for n, file in enumerate(dirs):
        subFile = os.path.join(path, file)
        if os.path.isdir(subFile):
            fileList.extend(listFiles(subFile, numPerDir))
        else:
            fileList.append(subFile)
    return fileList


def resize_and_crop(img_path, size, crop_type='middle'):
    """
    Resize and crop an image to fit the specified size.
    Parameters
    ----------
    img_path: path for the image to resize.
    modified_path: path to store the modified image.
    size: `(width, height)` tuple.
    crop_type: can be 'top', 'middle' or 'bottom', depending on this
        value, the image will cropped getting the 'top/left', 'midle' or
        'bottom/rigth' of the image to fit the size.
    raises:
    Exception: if can not open the file in img_path of there is problems
        to save the image.
    ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], size[0] * img.size[1] / img.size[0]), Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, (img.size[1] - size[1]) / 2, img.size[0], (img.size[1] + size[1]) / 2)
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((size[1] * img.size[0] / img.size[1], size[1]), Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = ((img.size[0] - size[0]) / 2, 0, (img.size[0] + size[0]) / 2, img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((size[0], size[1]), Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
    img = np.asarray(img)
    if len(img.shape) == 3:  # RGB
        return img.transpose((2, 0, 1))
    elif len(img.shape) == 2:  # grey
        return np.array([img] * 3)
    else:
        raise ValueError('ERROR: dim neither 3 nor 2')


# 数据格式为2D矩阵（样本数，图像行数*图像列数）
def mnist(onehot=False):
    data_dir = os.path.join(datasets_dir, 'mnist')

    def load_mnist_images(filename):
        filename = os.path.join(data_dir, filename)
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def load_mnist_labels(filename):
        filename = os.path.join(data_dir, filename)
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    tr_X = load_mnist_images('train-images.idx3-ubyte')
    tr_y = load_mnist_labels('train-labels.idx1-ubyte')
    te_X = load_mnist_images('t10k-images.idx3-ubyte')
    te_y = load_mnist_labels('t10k-labels.idx1-ubyte')
    if onehot:
        tr_y = one_hot(tr_y, 10)
        te_y = one_hot(te_y, 10)
    else:
        tr_y = np.asarray(tr_y)
        te_y = np.asarray(te_y)
    return tr_X, te_X, tr_y, te_y


def mnist_vary(name, onehot=False):
    data_dir = os.path.join(datasets_dir, 'mnist', 'vary')
    if name == 'basic':
        train_file = os.path.join(data_dir, 'mnist', 'train.mat')
        test_file = os.path.join(data_dir, 'mnist', 'test.mat')
    elif name == 'rotation':
        train_file = os.path.join(data_dir, 'mnist_rotation_new', 'train.mat')
        test_file = os.path.join(data_dir, 'mnist_rotation_new', 'test.mat')
    elif name == 'backrand':
        train_file = os.path.join(data_dir, 'mnist_background_random', 'train.mat')
        test_file = os.path.join(data_dir, 'mnist_background_random', 'test.mat')
    elif name == 'backimage':
        train_file = os.path.join(data_dir, 'mnist_background_images', 'train.mat')
        test_file = os.path.join(data_dir, 'mnist_background_images', 'test.mat')
    elif name == 'backimagerotation':
        train_file = os.path.join(data_dir, 'mnist_rotation_back_image_new', 'train.mat')
        test_file = os.path.join(data_dir, 'mnist_rotation_back_image_new', 'test.mat')
    else:
        raise ValueError('not found')
    tr_data = sio.loadmat(train_file)
    te_data = sio.loadmat(test_file)
    tr_X = tr_data['train'][:, :-1].reshape(-1, 1, 28, 28)
    tr_y = tr_data['train'][:, -1]
    te_X = te_data['test'][:, :-1].reshape(-1, 1, 28, 28)
    te_y = te_data['test'][:, -1]
    if onehot:
        tr_y = one_hot(tr_y, 10)
        te_y = one_hot(te_y, 10)
    else:
        tr_y = np.asarray(tr_y)
        te_y = np.asarray(te_y)
    return tr_X, te_X, tr_y, te_y


# 数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
def cifar(onehot=False):
    data_dir = os.path.join(datasets_dir, 'cifar10', 'cifar-10-batches-py')
    allFiles = os.listdir(data_dir)
    trFiles = [f for f in allFiles if f.startswith('data_batch')]
    tr_X = []
    tr_y = []
    for file in trFiles:
        fd = open(os.path.join(data_dir, file))
        dict = cPickle.load(fd)
        batchData = dict['data'].reshape(-1, 3, 32, 32)
        batchLabel = dict['labels']
        tr_X.append(batchData)
        tr_y.extend(batchLabel)
        fd.close()
    tr_X = np.vstack(tr_X)
    teFiles = [f for f in allFiles if f.find('test_batch') != -1]
    te_X = []
    te_y = []
    for file in teFiles:
        fd = open(os.path.join(data_dir, file))
        dict = cPickle.load(fd)
        batchData = dict['data'].reshape(-1, 3, 32, 32)
        batchLabel = dict['labels']
        te_X.append(batchData)
        te_y.extend(batchLabel)
        fd.close()
    te_X = np.vstack(te_X)
    if onehot:
        tr_y = one_hot(tr_y, 10)
        te_y = one_hot(te_y, 10)
    else:
        tr_y = np.asarray(tr_y)
        te_y = np.asarray(te_y)
    return tr_X, te_X, tr_y, te_y


def stl10(onehot=False):
    data_dir = os.path.join(datasets_dir, 'stl10_binary')

    def load_stl10_images(filename):
        filename = os.path.join(data_dir, filename)
        with open(filename, 'rb') as f:
            data = np.fromfile(f, np.uint8)
        data = data.reshape(-1, 3, 96, 96)
        return data

    def load_stl10_labels(filename):
        filename = os.path.join(data_dir, filename)
        with open(filename, 'rb') as f:
            data = np.fromfile(f, np.uint8)
        return data

    tr_X = load_stl10_images('train_X.bin')
    tr_y = load_stl10_labels('train_y.bin') - 1
    te_X = load_stl10_images('test_X.bin')
    te_y = load_stl10_labels('test_y.bin') - 1
    un_X = load_stl10_images('unlabeled_X.bin')
    if onehot:
        tr_y = one_hot(tr_y, 10)
        te_y = one_hot(te_y, 10)
    return tr_X, te_X, tr_y, te_y, un_X


def orl(onehot=False):
    data_dir = os.path.join(datasets_dir, 'orl_faces')

    def load_orl_images(dirname):
        images = []
        for i in xrange(10):
            filename = os.path.join(data_dir, dirname, str(i + 1) + '.pgm')
            with open(filename, 'rb') as f:
                data = np.fromfile(f, np.uint8)
            images.append(data[-10304:])
        images = np.array(images).reshape(10, 1, 112, 92)
        return images

    X = []
    for i in xrange(40):
        images = load_orl_images('s' + str(i + 1))
        X.append(images)
    X = np.concatenate(X, axis=0)
    y = np.repeat(np.arange(40), repeats=10)
    randindex = np.random.permutation(400)
    X = X[randindex]
    y = y[randindex]
    if onehot:
        y = one_hot(y, 40)
    return X, y


def yale(onehot=False):
    data_dir = os.path.join(datasets_dir, 'yalefaces', 'yalefaces')
    allfiles = listFiles(data_dir)
    allfiles = filter(lambda x: '.txt' not in x, allfiles)
    X = []
    y = []
    for filename in allfiles:
        X.append(imread(filename))
        pos = filename.find('subject') + 7
        y.append(int(filename[pos:pos + 2]))
    X = np.array(X).reshape(-1, 1, 243, 320)
    y = np.array(y) - 1
    randindex = np.random.permutation(X.shape[0])
    X = X[randindex]
    y = y[randindex]
    if onehot:
        y = one_hot(y, 15)
    return X, y


def yale_cropped(onehot=False):
    data_dir = os.path.join(datasets_dir, 'CroppedYale')

    def load_yale_images(dirname):
        file_dir = os.path.join(data_dir, dirname)
        allfiles = listFiles(file_dir)
        allfiles = filter(lambda x: '.pgm' in x and 'Ambient' not in x, allfiles)
        images = []
        for filename in allfiles:
            with open(filename, 'rb') as f:
                data = np.fromfile(f, np.uint8)
            images.append(data[-32256:])
        images = np.array(images).reshape((-1, 1, 192, 168))
        return images

    X = []
    for i in xrange(1, 40):
        num = str(i) if i > 9 else '0' + str(i)
        images = load_yale_images('yaleB' + num)
        X.append(images)
    X = np.concatenate(X, axis=0)
    y = np.repeat(np.arange(39), repeats=64)
    randindex = np.random.permutation(X.shape[0])
    X = X[randindex]
    y = y[randindex]
    if onehot:
        y = one_hot(y, 39)
    return X, y


def yale_extend(onehot=False):
    data_dir = os.path.join(datasets_dir, 'ExtendedYaleB')

    def load_yale_images(dirname):
        file_dir = os.path.join(data_dir, dirname)
        allfiles = listFiles(file_dir)
        allfiles = filter(lambda x: '.pgm' in x, allfiles)
        images = []
        for filename in allfiles:
            with open(filename, 'rb') as f:
                data = np.fromfile(f, np.uint8)
            images.append(data[-307200:])
        images = np.array(images).reshape((-1, 1, 480, 640))
        return images

    X = []
    for i in xrange(11, 40):
        images = load_yale_images('yaleB' + str(i))
        X.append(images)
    X = np.concatenate(X, axis=0)
    y = np.repeat(np.arange(29), repeats=585)
    randindex = np.random.permutation(X.shape[0])
    X = X[randindex]
    y = y[randindex]
    if onehot:
        y = one_hot(y, 29)
    return X, y


def _caltech_pre(which='101'):
    data_dir = os.path.join(datasets_dir, which + '_ObjectCategories')
    size = (112, 112)

    def load_caltech_images(dirname):
        file_dir = os.path.join(data_dir, dirname)
        allfiles = listFiles(file_dir)
        allfiles = filter(lambda x: '.jpg' in x, allfiles)
        images = []
        for filename in allfiles:
            data = resize_and_crop(filename, size)
            images.append(data)
        images = np.array(images).reshape((-1, 3) + size)
        return images

    X = []
    repeats = []
    categories = os.listdir(data_dir)
    n_class = len(categories)
    for dirname in categories:
        images = load_caltech_images(dirname)
        X.append(images)
        repeats.append(images.shape[0])
    X = np.concatenate(X, axis=0)
    y = np.repeat(np.arange(n_class), repeats=repeats)
    Xfile = os.path.join(datasets_dir, 'caltech', 'X' + which + '.npy')
    np.save(Xfile, X)
    yfile = os.path.join(datasets_dir, 'caltech', 'y' + which + '.npy')
    np.save(yfile, y)


def caltech(which='101', n_train=15, onehot=False):
    Xfile = os.path.join(datasets_dir, 'caltech', 'X' + which + '.npy')
    X = np.load(Xfile)
    yfile = os.path.join(datasets_dir, 'caltech', 'y' + which + '.npy')
    y = np.load(yfile)
    tr_X = []
    tr_y = []
    te_X = []
    te_y = []
    for label in xrange(int(which) + 1):
        labelindex = np.where(y == label)[0]
        count = len(labelindex)
        assert count > n_train
        randselect = np.random.permutation(count)
        tr_X.append(X[labelindex[randselect[:n_train]]])
        tr_y.append(y[labelindex[:n_train]])
        te_X.append(X[labelindex[randselect[n_train:]]])
        te_y.append(y[labelindex[n_train:]])
    tr_X = np.concatenate(tr_X, axis=0)
    tr_y = np.concatenate(tr_y, axis=0)
    te_X = np.concatenate(te_X, axis=0)
    te_y = np.concatenate(te_y, axis=0)
    if onehot:
        tr_y = one_hot(tr_y, int(which) + 1)
        te_y = one_hot(te_y, int(which) + 1)
    return tr_X, te_X, tr_y, te_y
