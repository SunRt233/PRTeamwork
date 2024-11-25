#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import pandas as pd


class DataContainer:
    def __init__(self, samples: list, labels: list):
        if len(samples) != len(labels):
            raise ValueError('样本的数量和样本标签的数量必须一致！')
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        s = str([{'sample': self.samples[i], 'label': self.labels[i]} for i in range(0,len(self.samples))])
        return s


class DataProvider:
    def __init__(self, store_path=r'./data'):
        self.store_path = store_path
        self._data: dict[str, list] = {}

    def __str__(self) -> str:
        s = str(self._data)
        return s

    def read_data(self):
        df = pd.read_excel(self.store_path, 'Data')
        # 之前采用PCA时没有注意到索引范围有误，导致特征选择出错
        self._data['samples'] = df.iloc[:, 5:27].values[1:]
        self._data['labels'] = df[['NSP']].values[1:]

    def provide_split_data(self, ratio=0.6, shuffle=False) -> tuple[DataContainer, DataContainer]:
        raw_data = self._data
        length = len(raw_data['samples'])

        data:list[tuple] = [(raw_data['samples'][i], raw_data['labels'][i]) for i in range(0, length)]

        if shuffle:
            random.shuffle(data)

        samples_shuffled = [item[0] for item in data]
        labels_shuffled =[item[1] for item in data]

        sp_point = int(len(samples_shuffled) * ratio)

        training_samples = samples_shuffled[:sp_point]
        training_labels = labels_shuffled[:sp_point]
        testing_samples = samples_shuffled[sp_point:]
        testing_labels = labels_shuffled[sp_point:]

        training_data = DataContainer(training_samples, training_labels)
        testing_data = DataContainer(testing_samples, testing_labels)

        return training_data, testing_data
