#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import numpy as np
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
        from numpy.ma.core import shape
        if len(samples) != len(labels):
            raise ValueError('样本的数量和样本标签的数量必须一致！')
        self.samples = samples  # 样本数据
        self.labels = labels  # 标签
        self._num_samples = len(samples)
        self._num_features = len(samples[0])
        self._num_labels = len(labels)
        self._num_classes = len(np.unique(self.labels))

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

    def num_samples(self):
        return self._num_samples

    def num_features(self):
        return self._num_features

    def num_labels(self):
        return self._num_labels

    def num_classes(self):
        return self._num_classes

    def classes(self):
        return np.unique(self.labels)


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

    def load_data(self):
        """
        读取数据。
        """
        df = pd.read_excel(self.store_path, 'Data')
        # 之前采用PCA时没有注意到索引范围有误，导致特征选择出错
        self._data['samples'] = df.iloc[:, 5:27].values[1:]  # 样本数据
        self._data['labels'] = df[['NSP']].values[1:]  # 标签

    def provide_data(self) -> DataContainer:
        """
        提供全部数据。
        :return: 全部数据的 DataContainer 对象
        """
        raw_data = self._data
        length = len(raw_data['samples'])

        samples = [sample for sample in raw_data['samples']]
        import numpy as np
        labels = [np.ravel(label) for label in raw_data['labels']]

        return DataContainer(samples, labels)

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

    def provide_balanced_split_data(self, train_ratio=0.6, verification_ratio=0.5, shuffle=False) -> tuple[DataContainer, DataContainer, DataContainer]:
        """
        提供按比例分割的训练数据和验证数据。
        :param train_ratio: 训练集占总数据的比例，默认为 0.6
        :param verification_ratio: 验证集占非训练数据的比例，默认为 0.5
        :param shuffle: 是否打乱数据，默认为 False
        :return: 训练数据和验证数据的 DataContainer 对象
        """
        # 获取数据集
        data = self.provide_data()

        # 将数据集按标签分类存储
        samples_dict = {int(label): [data.samples[i] for i in range(0, data.num_samples()) if data.labels[i] == label]
                        for label in data.labels}

        # 初始化训练集、验证集和测试集的列表
        train_pairs = []
        verification_pairs = []
        test_data_pairs = []

        # 遍历每个标签对应的样本
        for label in samples_dict.keys():
            samples = samples_dict[label]

            # 如果需要打乱数据，则执行打乱操作
            if shuffle:
                random.shuffle(samples)

            # 计算训练集和验证集的分割点
            tr_sp_point = int(len(samples) * train_ratio)
            ve_sp_point = int(len(samples) * (train_ratio + verification_ratio))

            # 根据分割点将数据添加到对应的集合中
            train_pairs.extend([(sample, label) for sample in samples[:tr_sp_point]])
            verification_pairs.extend([(sample, label) for sample in samples[tr_sp_point:ve_sp_point]])
            test_data_pairs.extend([(sample, label) for sample in samples[ve_sp_point:]])

        # 将数据对列表转换为 DataContainer 对象
        train_data = DataContainer([item[0] for item in train_pairs], [item[1] for item in train_pairs])
        verification_data = DataContainer([item[0] for item in verification_pairs],
                                          [item[1] for item in verification_pairs])
        test_data = DataContainer([item[0] for item in test_data_pairs], [item[1] for item in test_data_pairs])

        # 返回分割后的数据集
        return train_data, verification_data, test_data

