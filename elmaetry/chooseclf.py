# coding:utf-8
import gc
import numpy as np
from numpy.linalg import solve
from scipy.linalg import orth
from copy import copy, deepcopy
import myUtils
import cPickle
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def compute_beta_reg(Hmat, Tmat, C):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.eye(rows) / C + np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.eye(cols) / C + np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
    return beta


def compute_beta_rand(Hmat, Tmat, C):
    Crand = abs(np.random.uniform(0.1, 1.1)) * C
    return compute_beta_reg(Hmat, Tmat, Crand)


def compute_beta_direct(Hmat, Tmat):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
    return beta


def orthonormalize(filters):
    ndim = filters.ndim
    if ndim != 2:
        filters = np.expand_dims(filters, axis=0)
    rows, cols = filters.shape
    if rows >= cols:
        orthonormal = orth(filters)
    else:
        orthonormal = orth(filters.T).T
    if ndim != 2:
        orthonormal = np.squeeze(orthonormal, axis=0)
    return orthonormal


# 随机投影矩阵不同于一般的BP网络的初始化,要保持和输入一样的单位方差
def normal_random(input_unit, hidden_unit):
    std = 1.
    return np.random.normal(loc=0, scale=std, size=(input_unit, hidden_unit)), \
           np.random.normal(loc=0, scale=std, size=hidden_unit)


def uniform_random(input_unit, hidden_unit):
    ranges = 1.
    return np.random.uniform(low=-ranges, high=ranges, size=(input_unit, hidden_unit)), \
           np.random.uniform(low=-ranges, high=ranges, size=hidden_unit)


def relu(X):
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        xtmp = X[start:end]
        X[start:end] = 0.5 * (xtmp + abs(xtmp))
    return X


class CVInner(object):
    def get_train_acc(self, inputX, inputy):
        raise NotImplementedError

    def get_test_acc(self, inputX, inputy):
        raise NotImplementedError


class CVOuter(object):
    def train_cv(self, inputX, inputy):
        raise NotImplementedError

    def test_cv(self, inputX, inputy):
        raise NotImplementedError


def accuracy(ypred, ytrue):
    if ypred.ndim == 2:
        ypred = np.argmax(ypred, axis=1)
    if ytrue.ndim == 2:
        ytrue = np.argmax(ytrue, axis=1)
    return np.mean(ypred == ytrue)


class Classifier_ELM():
    def __init__(self, C, n_times):
        self.C = C
        self.n_times = n_times

    def get_train_output_for(self, inputX, inputy=None):
        n_hidden = int(self.n_times * inputX.shape[1])
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        self.beta = compute_beta_rand(H, inputy, self.C)
        out = np.dot(H, self.beta)
        return out

    def get_test_output_for(self, inputX):
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        out = np.dot(H, self.beta)
        return out


class Classifier_ELMcv(CVInner):
    def __init__(self, C_range, n_times):
        self.C_range = C_range
        self.n_times = n_times

    def get_train_acc(self, inputX, inputy):
        n_hidden = int(self.n_times * inputX.shape[1])
        print 'hiddens =', n_hidden
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        rows, cols = H.shape
        K = np.dot(H, H.T) if rows <= cols else np.dot(H.T, H)
        self.beta_list = []
        optacc = 0.
        optC = None
        for C in self.C_range:
            Crand = abs(np.random.uniform(0.1, 1.1)) * C
            beta = np.dot(H.T, solve(np.eye(rows) / Crand + K, inputy)) if rows <= cols \
                else solve(np.eye(cols) / Crand + K, np.dot(H.T, inputy))
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            self.beta_list.append(copy(beta))
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc

    def get_test_acc(self, inputX, inputy):
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        optacc = 0.
        optC = None
        for beta, C in zip(self.beta_list, self.C_range):
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc


class Classifier_OSELM():
    def __init__(self, C, n_times):
        self.C = C
        self.n_times = n_times

    def _initial(self, inputX0, inputy0):
        H0 = np.dot(inputX0, self.W) + self.b
        rows, cols = H0.shape
        assert rows >= cols  # 只允许行大于列
        del inputX0
        H0 = relu(H0)
        self.K = H0.T.dot(H0) + np.eye(cols) / self.C  # 只在这里加入一次惩罚项
        self.beta = solve(self.K, H0.T.dot(inputy0))
        return H0

    def _sequential(self, inputX, inputy):
        H1 = np.dot(inputX, self.W) + self.b
        del inputX
        H1 = relu(H1)
        self.K = self.K + H1.T.dot(H1)  # 惩罚项np.eye(cols) / self.C在这里不再加入
        self.beta = self.beta + solve(self.K, H1.T.dot(inputy - H1.dot(self.beta)))
        return H1

    def get_train_output_for(self, inputX, inputy=None, bias_scale=25):
        n_sample, n_feature = inputX.shape
        n_hidden = int(self.n_times * n_feature)
        assert n_sample >= n_hidden
        # 固定输入随机矩阵
        self.W, self.b = normal_random(input_unit=n_feature, hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        batchSize = int(round(float(n_sample) / 5))
        if batchSize < n_hidden: batchSize = n_hidden
        splits = int(np.ceil(float(n_sample) / batchSize))
        H = []
        for i in xrange(splits):  # 分为splits次训练
            inputXtmp, inputytmp = inputX[:batchSize], inputy[:batchSize]
            inputX, inputy = inputX[batchSize:], inputy[batchSize:]
            Htmp = self._initial(inputXtmp, inputytmp) if i == 0 else self._sequential(inputXtmp, inputytmp)
            H = np.concatenate([H, Htmp], axis=0) if len(H) != 0 else Htmp
        out = np.dot(H, self.beta)
        return out

    def get_test_output_for(self, inputX):
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        out = np.dot(H, self.beta)
        return out


class Classifier_OSELMcv(CVInner):
    def __init__(self, C_range, n_times):
        self.C_range = C_range
        self.n_times = n_times

    def _initial(self, inputX0, inputy0):
        H0 = np.dot(inputX0, self.W) + self.b
        rows, cols = H0.shape
        assert rows >= cols  # 只允许行大于列
        del inputX0
        H0 = relu(H0)
        self.K = H0.T.dot(H0)  # 所有的K只有第一次加的C不同
        self.beta_list = []
        for C in self.C_range:
            beta = solve(self.K + np.eye(cols) / C, H0.T.dot(inputy0))
            self.beta_list.append(copy(beta))
        return H0

    def _sequential(self, inputX, inputy):
        H1 = np.dot(inputX, self.W) + self.b
        rows, cols = H1.shape
        del inputX
        H1 = relu(H1)
        self.K += H1.T.dot(H1)
        for C in self.C_range:
            beta = self.beta_list.pop(0)
            beta += solve(self.K + np.eye(cols) / C, H1.T.dot(inputy - H1.dot(beta)))
            self.beta_list.append(copy(beta))
        return H1

    def get_train_acc(self, inputX, inputy):
        n_sample, n_feature = inputX.shape
        n_hidden = int(self.n_times * n_feature)
        assert n_sample >= n_hidden
        inputy_copy = np.copy(inputy)
        # 固定输入随机矩阵
        self.W, self.b = normal_random(input_unit=n_feature, hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        batchSize = int(round(float(n_sample) / 5))
        if batchSize < n_hidden: batchSize = n_hidden
        splits = int(np.ceil(float(n_sample) / batchSize))
        H = []
        for i in xrange(splits):  # 分为splits次训练
            inputXtmp, inputytmp = inputX[:batchSize], inputy_copy[:batchSize]
            inputX, inputy_copy = inputX[batchSize:], inputy_copy[batchSize:]
            Htmp = self._initial(inputXtmp, inputytmp) if i == 0 else self._sequential(inputXtmp, inputytmp)
            H = np.concatenate([H, Htmp], axis=0) if len(H) != 0 else Htmp
        del self.K
        optacc = 0.
        optC = None
        for C, beta in zip(self.C_range, self.beta_list):
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc

    def get_test_acc(self, inputX, inputy):
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        optacc = 0.
        optC = None
        for C, beta in zip(self.C_range, self.beta_list):
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc


class Classifier_ELMtimescv(CVOuter):
    def __init__(self, elm_type, n_rep, C_range, times_range):
        elm_dict = {'elm': Classifier_ELMcv, 'oselm': Classifier_OSELMcv}
        if elm_type not in elm_dict.keys():
            raise NotImplemented
        self.clf_class = elm_dict[elm_type]
        self.C_range = C_range
        self.n_rep = n_rep
        self.times_range = times_range
        self.clf_list = []

    def train_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        for n_times in self.times_range:
            print 'times', n_times, ':'
            for j in xrange(self.n_rep):
                print 'repeat', j
                clf = self.clf_class(self.C_range, n_times)
                C, acc = clf.get_train_acc(inputX, inputy)
                self.clf_list.append(deepcopy(clf))
                if acc > optacc:
                    optacc = acc
                    optC = C
            print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        for clf in self.clf_list:
            print 'times', clf.n_times, ':'
            C, acc = clf.get_test_acc(inputX, inputy)
            if acc > optacc:
                optacc = acc
                optC = C
            print 'test opt', optC, optacc


def addtrans_decomp(X, Y=None):
    if Y is None: Y = X
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        Xtmp = X[start:end, :] + Y[:, start:end].T
        result = np.concatenate([result, Xtmp], axis=0) if len(result) != 0 else Xtmp
    return result


def kernel(Xtr, Xte=None, kernel_type='rbf', kernel_args=(1.,)):
    rows_tr = Xtr.shape[0]
    if not isinstance(kernel_args, (tuple, list)): kernel_args = (kernel_args,)
    if kernel_type == 'rbf':
        if Xte is None:
            H = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            # omega = H + H.T - 2 * np.dot(Xtr, Xtr.T)
            omega = addtrans_decomp(H) - 2 * np.dot(Xtr, Xtr.T)
            del H, Xtr
            omega = np.exp(-omega / kernel_args[0])
        else:
            rows_te = Xte.shape[0]
            Htr = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_te, axis=1)
            Hte = np.repeat(np.sum(Xte ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            # omega = Htr + Hte.T - 2 * np.dot(Xtr, Xte.T)
            omega = addtrans_decomp(Htr, Hte) - 2 * np.dot(Xtr, Xte.T)
            del Htr, Hte, Xtr, Xte
            omega = np.exp(-omega / kernel_args[0])
    elif kernel_type == 'lin':
        if Xte is None:
            omega = np.dot(Xtr, Xtr.T)
        else:
            omega = np.dot(Xtr, Xte.T)
    elif kernel_type == 'poly':
        if Xte is None:
            omega = (np.dot(Xtr, Xtr.T) + kernel_args[0]) ** kernel_args[1]
        else:
            omega = (np.dot(Xtr, Xte.T) + kernel_args[0]) ** kernel_args[1]
    elif kernel_type == 'wav':
        if Xte is None:
            H = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            omega = H + H.T - 2 * np.dot(Xtr, Xtr.T)
            H1 = np.repeat(np.sum(Xtr, axis=1, keepdims=True), rows_tr, axis=1)
            omega1 = H1 - H1.T
            omega = np.cos(omega1 / kernel_args[0]) * np.exp(-omega / kernel_args[1])
        else:
            rows_te = Xte.shape[0]
            Htr = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_te, axis=1)
            Hte = np.repeat(np.sum(Xte ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            omega = Htr + Hte.T - 2 * np.dot(Xtr, Xte.T)
            Htr1 = np.repeat(np.sum(Xtr, axis=1, keepdims=True), rows_te, axis=1)
            Hte1 = np.repeat(np.sum(Xte, axis=1, keepdims=True), rows_tr, axis=1)
            omega1 = Htr1 - Hte1.T
            omega = np.cos(omega1 / kernel_args[0]) * np.exp(-omega / kernel_args[1])
    else:
        raise NotImplemented
    return omega


class Classifier_KELM():
    def __init__(self, C, kernel_type, kernel_args):
        self.C = C
        self.kernel_type = kernel_type
        self.kernel_args = kernel_args

    def get_train_output_for(self, inputX, inputy=None):
        self.trainX = inputX
        omega = kernel(inputX, self.kernel_type, self.kernel_args)
        rows = omega.shape[0]
        Crand = abs(np.random.uniform(0.1, 1.1)) * self.C
        self.beta = solve(np.eye(rows) / Crand + omega, inputy)
        out = np.dot(omega, self.beta)
        return out

    def get_test_output_for(self, inputX):
        omega = kernel(self.trainX, inputX, self.kernel_type, self.kernel_args)
        del inputX
        out = np.dot(omega.T, self.beta)
        return out


class Classifier_KELMcv(CVOuter):
    def __init__(self, C_range, kernel_type, kernel_args_list):
        self.C_range = C_range
        self.kernel_type = kernel_type
        self.kernel_args_list = kernel_args_list

    def train_cv(self, inputX, inputy):
        self.trainX = inputX
        self.beta_list = []
        optacc = 0.
        optC = None
        optarg = None
        for kernel_args in self.kernel_args_list:
            omega = kernel(inputX, None, self.kernel_type, kernel_args)
            rows = omega.shape[0]
            for C in self.C_range:
                Crand = abs(np.random.uniform(0.1, 1.1)) * C
                beta = solve(np.eye(rows) / Crand + omega, inputy)
                out = np.dot(omega, beta)
                acc = accuracy(out, inputy)
                self.beta_list.append(copy(beta))
                print '\t', kernel_args, C, acc
                if acc > optacc:
                    optacc = acc
                    optC = C
                    optarg = kernel_args
            del omega
            gc.collect()
        print 'train opt', optarg, optC, optacc

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        optarg = None
        num = 0
        for kernel_args in self.kernel_args_list:
            omega = kernel(self.trainX, inputX, self.kernel_type, kernel_args)
            for C in self.C_range:
                out = np.dot(omega.T, self.beta_list[num])
                acc = accuracy(out, inputy)
                print '\t', kernel_args, C, acc
                num += 1
                if acc > optacc:
                    optacc = acc
                    optC = C
                    optarg = kernel_args
            del omega
            gc.collect()
        print 'test opt', optarg, optC, optacc


class Classifier_kNN(CVInner):
    def __init__(self, k, w):
        self.k = k
        self.w = w

    def get_train_acc(self, inputX, inputy):
        self.clf = KNeighborsClassifier(n_neighbors=self.k, weights=self.w, n_jobs=-1)
        self.clf.fit(inputX, inputy)
        return self.clf.score(inputX, inputy)

    def get_test_acc(self, inputX, inputy):
        return self.clf.score(inputX, inputy)


class Classifier_kNNcv(CVOuter):
    def __init__(self, k_range, w_list):
        self.k_range = k_range
        self.w_list = w_list

    def train_cv(self, inputX, inputy):
        optacc = 0.
        optk = None
        optw = None
        self.clf_list = []
        for w in self.w_list:
            for k in self.k_range:
                clf = Classifier_kNN(k, w)
                acc = clf.get_train_acc(inputX, inputy)
                self.clf_list.append(deepcopy(clf))
                print '\t', w, k, acc
                if acc > optacc:
                    optacc = acc
                    optk = k
                    optw = w
        print 'train opt', optw, optk, optacc

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optk = None
        optw = None
        for clf in self.clf_list:
            acc = clf.get_test_acc(inputX, inputy)
            print '\t', clf.w, clf.k, acc
            if acc > optacc:
                optacc = acc
                optk = clf.k
                optw = clf.w
        print 'test opt', optw, optk, optacc


class Classifier_SVM(CVInner):
    def __init__(self, C):
        self.C = C

    def get_train_acc(self, inputX, inputy):
        dual = inputX.shape[0] < inputX.shape[1]
        self.clf = LinearSVC(C=self.C, dual=dual, max_iter=5000)
        self.clf.fit(inputX, inputy)
        return self.clf.score(inputX, inputy)

    def get_test_acc(self, inputX, inputy=None):
        return self.clf.score(inputX, inputy)


class Classifier_SVMcv(CVOuter):
    def __init__(self, C_range):
        self.C_range = C_range

    def train_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        self.clf_list = []
        for C in self.C_range:
            clf = Classifier_SVM(C)
            acc = clf.get_train_acc(inputX, inputy)
            self.clf_list.append(deepcopy(clf))
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        for clf in self.clf_list:
            acc = clf.get_test_acc(inputX, inputy)
            print '\t', clf.C, acc
            if acc > optacc:
                optacc = acc
                optC = clf.C
        print 'test opt', optC, optacc


class Selector():
    def __init__(self, clf_type, clf_args):
        clf_dict = {'elm': Classifier_ELMtimescv, 'kelm': Classifier_KELMcv}
        if clf_type not in clf_dict.keys():
            raise NotImplemented
        self.clf = clf_dict[clf_type](**clf_args)

    def train(self, inputX, inputy):
        self.clf.train_cv(inputX, inputy)

    def test(self, inputX, inputy):
        self.clf.test_cv(inputX, inputy)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    del tr_X, te_X
    tr_X = cPickle.load(open('/home/zhufenghao/trainfeat.pkl', 'r'))
    te_X = cPickle.load(open('/home/zhufenghao/testfeat.pkl', 'r'))
    kelm_args = {'C_range': 10 ** np.arange(0, 5., 1.), 'kernel_type': 'rbf',
                 'kernel_args_list': 10 ** np.arange(0., 5., 1.)}
    selector = Selector('kelm', kelm_args)
    selector.train(tr_X, tr_y)
    selector.test(te_X, te_y)
    elm_args1 = {'elm_type': 'elm', 'n_rep': 2, 'C_range': 10 ** np.arange(-1., 3., 1.), 'times_range': [12, ]}
    selector = Selector('elm', elm_args1)
    selector.train(tr_X, tr_y)
    selector.test(te_X, te_y)
    elm_args2 = {'elm_type': 'oselm', 'n_rep': 2, 'C_range': 10 ** np.arange(-1., 3., 1.), 'times_range': [12, ]}
    selector = Selector('elm', elm_args2)
    selector.train(tr_X, tr_y)
    selector.test(te_X, te_y)


if __name__ == '__main__':
    main()
