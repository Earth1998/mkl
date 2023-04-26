import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import *
from augmentationgc import Augmentation
import random


from config1 import config
from gcs import rs
import time

path = 'all_descriptorswithKI.xlsx'
IC50 = pd.read_excel(path, header=None)
IC50 = IC50.values
X = IC50[:, 1:]
print(X)
y = IC50[:, 0]
print(y)

aug = Augmentation(alpha=1.0, beta=1.0)

X1, y1 = aug.train(X, y)
Y = OneHotEncoder().fit_transform(y1.reshape(-1, 1)).toarray()
print('-'*50)
print('-'*50)
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
print(X1.shape)


acc_list = []
pre_list = []
rec_list = []
f1_list = []

se = []
sp = []

tpr_list = []
fpr_list = []

for i in range(2):
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        X_train_aug, y_train_aug = aug.train(X_train, y_train)

        X_test_aug, y_test_aug = aug.test(X_train, X_test, y_train, y_test)
        print(X_train_aug.shape)
        print(X_test_aug.shape)

        X_train_aug = scaler.transform(X_train_aug)
        X_test_aug = scaler.transform(X_test_aug)


        for m in range(len(y_train_aug)):
            if y_train_aug[m] == 0:
                y_train_aug[m] = -1
        for n in range(len(y_test_aug)):
            if y_test_aug[n] == 0:
                y_test_aug[n] = -1
        config['rs_v0'] = np.ones(shape=(X_train_aug.shape[0],), dtype=np.float64)
        config['rs_eta'] = 0.5
        model = rs(config)
        model.fit(X_train_aug, y_train_aug)
        y_pred = model.predict(X_test_aug, last_model_flag=True)
        print(y_pred.shape)
        # y_pred = np.mean(y_pred, axis=1)
        for j in range(len(y_pred)):
            if y_pred[j] < 0:
                y_pred[j] = -1
            else:
                y_pred[j] = 1
        print(y_pred.shape)

        acc = accuracy_score(y_test_aug, y_pred)
        pre = precision_score(y_test_aug, y_pred)
        rec = recall_score(y_test_aug, y_pred)
        f1 = f1_score(y_test_aug, y_pred)
        print('acc', acc)
        print('pre', pre)
        print('rec', rec)
        print('f1', f1)
        print('=' * 50)

        confusion = confusion_matrix(y_test_aug, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]

        # TPR = TP / (TP + FN)
        sensitivity = TP / (TP + FN)
        tpr = sensitivity
        # FPR = FP / (FP + TN)
        fpr = FP / (FP + TN)
        # TNR = TN / (FP + TN)
        specificity = TN / (FP + TN)

        se.append(sensitivity)
        sp.append(specificity)

        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

acc_list = np.array(acc_list)
pre_list = np.array(pre_list)
rec_list = np.array(rec_list)
f1_list = np.array(f1_list)

se = np.array(se)
sp = np.array(sp)

tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)

tpr_point = tpr_list.mean()
fpr_point = fpr_list.mean()

x_fpr = [0, fpr_point, 1]
y_tpr = [0, tpr_point, 1]

roc_auc = auc(x_fpr, y_tpr)

print('-' * 50)
print('tpr', tpr_point)
print('fpr', fpr_point)
print('roc_auc', roc_auc)
print('-' * 50)
print('acc', acc_list.mean())
print('pre', pre_list.mean())
print('rec', rec_list.mean())
print('f1', f1_list.mean())
print('std', acc_list.std())
print('sensitivity', se.mean())
print('specificity', sp.mean())
