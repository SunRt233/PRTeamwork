#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import pandas as pd


class DataContainer:
    """
    存储样本数据和对应的标签。
    """
    def __init__(self, samples: list, labels: list):
        """
        初始化 DataContainer。
        :param samples: 样本数据列表
        :param labels: 标签列表
        """
        if len(samples) != len(labels):
            raise ValueError('样本的数量和样本标签的数量必须一致！')
        self.samples = samples  # 样本数据
        self.labels = labels  # 标签

    def __len__(self):
        """
        返回样本数量。
        :return: 样本数量
        """
        return len(self.samples)

    def __str__(self):
        """
        返回 DataContainer 的字符串表示形式。
        :return: 字符串表示形式
        """
        s = str([{'sample': self.samples[i], 'label': self.labels[i]} for i in range(0, len(self.samples))])
        return s


class DataProvider:
    """
    提供数据读取和分割功能。
    """
    def __init__(self, store_path=r'./data'):
        """
        初始化 DataProvider。
        :param store_path: 数据存储路径，默认为 './data'
        """
        self.store_path = store_path  # 数据存储路径
        self._data: dict[str, list] = {}  # 存储读取的数据

    def __str__(self) -> str:
        """
        返回 DataProvider 的字符串表示形式。
        :return: 字符串表示形式
        """
        s = str(self._data)
        return s

    def read_data(self):
        """
        从 Excel 文件中读取数据。
        """
        df = pd.read_excel(self.store_path, 'Data')
        # 之前采用PCA时没有注意到索引范围有误，导致特征选择出错
        self._data['samples'] = df.iloc[:, 5:27].values[1:]  # 样本数据
        self._data['labels'] = df[['NSP']].values[1:]  # 标签

    def provide_split_data(self, ratio=0.6, shuffle=False) -> tuple[DataContainer, DataContainer]:
        """
        提供按比例分割的训练数据和测试数据。
        :param ratio: 训练集占总数据的比例，默认为 0.6
        :param shuffle: 是否打乱数据，默认为 False
        :return: 训练数据和测试数据的 DataContainer 对象
        """
        raw_data = self._data
        length = len(raw_data['samples'])

        data: list[tuple] = [(raw_data['samples'][i], raw_data['labels'][i]) for i in range(0, length)]

        if shuffle:
            random.shuffle(data)

        samples_shuffled = [item[0] for item in data]
        labels_shuffled = [item[1] for item in data]

        sp_point = int(len(samples_shuffled) * ratio)

        training_samples = samples_shuffled[:sp_point]
        training_labels = labels_shuffled[:sp_point]
        testing_samples = samples_shuffled[sp_point:]
        testing_labels = labels_shuffled[sp_point:]

        training_data = DataContainer(training_samples, training_labels)
        testing_data = DataContainer(testing_samples, testing_labels)

        return training_data, testing_data
