#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from math import sqrt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from data import DataContainer

import numpy as np

class Classifier:
    class Method(Enum):
        PCA = 1
        FLD = 2

    def __init__(self):
        self._pca_processed_data = None
        self._n = None
        self._trained_memory = None

    def train(self, training_data: DataContainer, method: Method, **train_args) -> None:
        match method:
            case Classifier.Method.PCA:
                self._store_memory(self._pca_process(training_data.samples, **train_args))
                knn_data = DataContainer(self._get_memory(), training_data.labels)
                self._knn_init(knn_data, **train_args)
                self._check_selected_method(method)
            case Classifier.Method.FLD:
                # self._fld_train(training_data, **train_args)
                pass
    def predict(self, data: list) -> list:
        match self.method:
            case Classifier.Method.PCA:
                return self._knn_predict(self._pca_process(data))
            case Classifier.Method.FLD:
                return self._knn_predict(self._fld_process(data))

    def _pca_process(self, data: list, **args) -> list:
        if ('dimens' in args) and (not hasattr(self,'dimens')):
            dimens = args['dimens']
            self.dimens = dimens
        elif hasattr(self,'dimens'):
            dimens = self.dimens
        else:
            pca_scaler = StandardScaler()
            pca = PCA()
            scaled_samples = pca_scaler.fit_transform(data)
            pca_processed_samples = pca.fit_transform(scaled_samples)
            print('主成分方差贡献率：',["{:.8f}".format(x) for x in pca.explained_variance_ratio_])
            temp = np.cumsum(pca.explained_variance_ratio_)
            dimens = 0
            for i in range(0,len(temp)):
                print(f'成分数：{i+1}, 累计方差贡献率：{temp[i]}')
                if temp[i] >= 0.9:
                    dimens = i+1
                    break
            print(f'选择保留主成分个数为：{dimens}')

            self.dimens = dimens


        pca_scaler = StandardScaler()
        pca = PCA(n_components=dimens)
        scaled_samples = pca_scaler.fit_transform(data)
        pca_processed_samples = pca.fit_transform(scaled_samples)
        return pca_processed_samples


    def _fld_process(self, data: list, **args) -> list:
        ...

    def _check_selected_method(self, method: Method):
        if hasattr(self, 'method'):
            if self.method != method:
                raise ValueError(f'当前已选择{self.method.name}作为训练方法，无法切换！')
        else:
            self.method = method

    def _knn_init(self, training_data: DataContainer, **train_args) -> None:
        self._n = int(min(35.0,sqrt(len(training_data.samples))))
        self._knn = KNeighborsClassifier(n_neighbors=self._n)
        self._knn.fit(training_data.samples, training_data.labels)

    def _knn_predict(self, data) -> list:
        return self._knn.predict(data)

    def _store_memory(self, memory):
        self._trained_memory = memory

    def _get_memory(self):
        if not hasattr(self, '_trained_memory'):
            raise ValueError('还没训练呢！')
        return self._trained_memory