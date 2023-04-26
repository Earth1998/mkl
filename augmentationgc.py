import numpy as np
from itertools import permutations
from scipy.special import gamma


class Augmentation():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def GC(self, x1, x2):
        s = 1 / np.power(self.beta, self.alpha)
        gammacoef = self.alpha / (2 * self.beta * gamma(1 / self.alpha))
        gc = gammacoef * np.exp(-s * np.abs(x1 - x2) ** self.alpha)
        return gc.mean()

    def train(self, X, y):
        N = X.shape[0]
        num_feature = X.shape[1]
        X_tran = np.zeros((N * (N - 1), num_feature))
        y_tran = np.zeros((N * (N - 1)))

        for i, p in enumerate(permutations(X, 2)):
            q = list(p)
            w = self.GC(q[0].copy(), q[1].copy())
            X_tran[i, :] = w * (q[0].copy() - q[1].copy())

            # X_tran[0, :, :, i] = (q[0].copy()).reshape((int_edge, int_edge))
            # X_tran[1, :, :, i] = (q[1].copy()).reshape((int_edge, int_edge))

        for i, p in enumerate(permutations(y, 2)):
            q = list(p)
            q.sort()
            for m in range(len(q)):
                for n in range(len(p)):
                    if q[m] == p[n]:
                        q[m] = n

            if q == [0, 1]:
                y_tran[i] = 0
            else:
                y_tran[i] = 1
        return X_tran, y_tran

    def test(self, X_train, X_test, y_train, y_test):
        N1 = X_train.shape[0]
        N2 = X_test.shape[0]
        num_feature = X_train.shape[1]
        X_tran = np.zeros((N1 * N2, num_feature))
        y_tran = np.zeros((N1 * N2))

        # X_tran = np.zeros((2, int_edge, int_edge, 2 * N1 * N2))
        # y_tran = np.zeros((2 * N1 * N2))

        for i in range(N1):
            for j in range(N2):
                w = self.GC(X_train[i], X_test[j])
                X_tran[i * N2 + j, :] = w * (X_train[i] - X_test[j])
                if y_train[i] < y_test[j]:
                    y_tran[i * N2 + j] = 0
                else:
                    y_tran[i * N2 + j] = 1

        return X_tran, y_tran

# def train_augmentation(X, y):
#     N = X.shape[0]
#     num_feature = X.shape[1]
#     X_tran = np.zeros((N * (N - 1), num_feature))
#     y_tran = np.zeros((N * (N - 1)))
#
#     for i, p in enumerate(permutations(X, 2)):
#         q = list(p)
#         X_tran[i, :] = q[0].copy() - q[1].copy()
#         # X_tran[0, :, :, i] = (q[0].copy()).reshape((int_edge, int_edge))
#         # X_tran[1, :, :, i] = (q[1].copy()).reshape((int_edge, int_edge))
#
#     for i, p in enumerate(permutations(y, 2)):
#         q = list(p)
#         q.sort()
#         for m in range(len(q)):
#             for n in range(len(p)):
#                 if q[m] == p[n]:
#                     q[m] = n
#
#         if q == [0, 1]:
#             y_tran[i] = 0
#         else:
#             y_tran[i] = 1
#     return X_tran, y_tran
#
#
# def test_augmentation(X_train, X_test, y_train, y_test):
#     N1 = X_train.shape[0]
#     N2 = X_test.shape[0]
#     num_feature = X_train.shape[1]
#     X_tran = np.zeros((N1 * N2, num_feature))
#     y_tran = np.zeros((N1 * N2))
#
#     # X_tran = np.zeros((2, int_edge, int_edge, 2 * N1 * N2))
#     # y_tran = np.zeros((2 * N1 * N2))
#
#     for i in range(N1):
#         for j in range(N2):
#             X_tran[i * N2 + j, :] = X_train[i] - X_test[j]
#             if y_train[i] < y_test[j]:
#                 y_tran[i * N2 + j] = 0
#             else:
#                 y_tran[i * N2 + j] = 1
#
#     return X_tran, y_tran
#
#
#
# def GC(x1, x2, alpha, beta):
#     s = 1 / np.power(beta, alpha)
#     gammacoef = alpha / (2 * beta * gamma(1 / alpha))
#     gc = gammacoef * np.exp(-s * np.abs(x1 - x2) ** alpha)
#     return gc.mean()
