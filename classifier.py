#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from math import sqrt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from data import DataContainer


class Memory:
    """
    存储处理后的数据及其处理器和维度信息。
    """
    def __init__(self, processed_data, processor, dimens: int):
        self.processed_data = processed_data  # 处理后的数据
        self.processor = processor  # 数据处理对象（PCA 或 LDA）
        self.dimens = dimens  # 保留的主成分个数


class Classifier:
    """
    分类器类，支持 PCA 和 LDA 两种降维方法。
    """
    class Method(Enum):
        """
        枚举类，定义分类器支持的方法。
        """
        PCA = 1  # 主成分分析
        LDA = 2  # 线性判别分析

    def __init__(self, print_log=False):
        """
        初始化分类器。
        :param print_log: 是否打印日志，默认为 False
        """
        self._pca_processed_data = None  # PCA 处理后的数据
        self._n = None  # KNN 的邻居数
        self._trained_memory = None  # 训练后的内存
        self.print_log = print_log  # 是否打印日志

    def train(self, training_data: DataContainer, method: Method, **train_args) -> None:
        """
        训练分类器。
        :param training_data: 训练数据容器
        :param method: 降维方法（PCA 或 LDA）
        :param train_args: 其他训练参数
        """
        match method:
            case Classifier.Method.PCA:
                self._store_memory(self._pca_train(training_data.samples, **train_args))
                knn_data = DataContainer(self._get_memory().processed_data, training_data.labels)
                self._knn_init(knn_data, **train_args)
                self._check_selected_method(method)
            case Classifier.Method.LDA:
                self._store_memory(self._lda_train(training_data.samples, training_data.labels, **train_args))
                knn_data = DataContainer(self._get_memory().processed_data, training_data.labels)
                self._knn_init(knn_data, **train_args)
                self._check_selected_method(method)

    def predict(self, data: list) -> list:
        """
        预测数据。
        :param data: 待预测的数据
        :return: 预测结果
        """
        match self.method:
            case Classifier.Method.PCA:
                return self._knn_predict(self._pca_process(data))
            case Classifier.Method.LDA:
                return self._knn_predict(self._lda_process(data))

    def _pca_train(self, data: list, **args) -> Memory:
        """
        使用 PCA 训练数据。
        :param data: 训练数据
        :param args: 其他参数
        :return: 训练后的内存对象
        """
        if ('dimens' in args) and (not hasattr(self, 'dimens')):
            dimens = args['dimens']
        else:
            pca_scaler = StandardScaler()
            pca = PCA()
            scaled_samples = pca_scaler.fit_transform(data)
            _ = pca.fit_transform(scaled_samples)
            if self.print_log:
                print('主成分方差贡献率：', ["{:.8f}".format(x) for x in pca.explained_variance_ratio_])
            temp = np.cumsum(pca.explained_variance_ratio_)
            dimens = 0
            for i in range(0, len(temp)):
                if temp[i] >= 0.9:
                    dimens = i + 1
                    break
            if self.print_log:
                print(f'选择保留主成分个数为：{dimens}')
        pca_scaler = StandardScaler()
        pca = PCA(n_components=dimens)
        scaled_samples = pca_scaler.fit_transform(data)
        pca_processed_samples = pca.fit_transform(scaled_samples)
        return Memory(pca_processed_samples, pca, dimens)

    def _pca_process(self, data: list, **args) -> list:
        """
        使用 PCA 处理数据。
        :param data: 待处理的数据
        :param args: 其他参数
        :return: 处理后的数据
        """
        pca = self._get_memory().processor
        scaler = StandardScaler()
        scaled_samples = scaler.fit_transform(data)
        pca_processed_samples = pca.transform(scaled_samples)
        return pca_processed_samples

    def _lda_train(self, data: list, labels: list, **args) -> Memory:
        """
        使用 LDA 训练数据。
        :param data: 训练数据
        :param labels: 训练标签
        :param args: 其他参数
        :return: 训练后的内存对象
        """
        if ('dimens' in args) and (not hasattr(self, 'dimens')):
            dimens = args['dimens']
        else:
            lda_scaler = StandardScaler()
            lda = LinearDiscriminantAnalysis()
            scaled_samples = lda_scaler.fit_transform(data)
            _ = lda.fit_transform(scaled_samples, np.ravel(labels))
            if self.print_log:
                print('主成分方差贡献率：', ["{:.8f}".format(x) for x in lda.explained_variance_ratio_])
            temp = np.cumsum(lda.explained_variance_ratio_)
            dimens = 0
            for i in range(0, len(temp)):
                if temp[i] >= 0.9:
                    dimens = i + 1
                    break
            if self.print_log:
                print(f'选择保留主成分个数为：{dimens}')
        lda_scaler = StandardScaler()
        lda = LinearDiscriminantAnalysis(n_components=dimens)
        scaled_samples = lda_scaler.fit_transform(data)
        lda.fit(scaled_samples, np.ravel(labels))
        lda_processed_samples = lda.transform(scaled_samples)
        return Memory(lda_processed_samples, lda, dimens)

    def _lda_process(self, data: list, **args) -> list:
        """
        使用 LDA 处理数据。
        :param data: 待处理的数据
        :param args: 其他参数
        :return: 处理后的数据
        """
        lda = self._get_memory().processor
        scaler = StandardScaler()
        scaled_samples = scaler.fit_transform(data)
        lda_processed_samples = lda.transform(scaled_samples)
        return lda_processed_samples

    def _check_selected_method(self, method: Method):
        """
        检查当前选择的降维方法是否与已选择的方法一致。
        :param method: 当前选择的降维方法
        """
        if hasattr(self, 'method'):
            if self.method != method:
                raise ValueError(f'当前已选择{self.method.name}作为训练方法，无法切换！')
        else:
            self.method = method

    def _knn_init(self, training_data: DataContainer, **train_args) -> None:
        """
        初始化 KNN 分类器。
        :param training_data: 训练数据容器
        :param train_args: 其他训练参数
        """
        self._n = int(min(35.0, sqrt(len(training_data.samples))))
        self._knn = KNeighborsClassifier(n_neighbors=self._n)
        self._knn.fit(training_data.samples, np.ravel(training_data.labels))

    def _knn_predict(self, data) -> list:
        """
        使用 KNN 分类器进行预测。
        :param data: 待预测的数据
        :return: 预测结果
        """
        return self._knn.predict(data)

    def _store_memory(self, memory: Memory):
        """
        存储训练后的内存对象。
        :param memory: 训练后的内存对象
        """
        self._trained_memory = memory

    def _get_memory(self) -> Memory:
        """
        获取训练后的内存对象。
        :return: 训练后的内存对象
        """
        if not hasattr(self, '_trained_memory'):
            raise ValueError('还没训练呢！')
        return self._trained_memory
