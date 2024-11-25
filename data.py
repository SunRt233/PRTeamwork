#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd


class DataProvider:
    def __init__(self, store_path=r'./data'):
        self.store_path = store_path
        self.data: dict[str, list] = {}

    def __str__(self) -> str:
        s = str(self.data)
        return s

    def read_data(self):
        df = pd.read_excel(self.store_path, 'Data')
        # 之前采用PCA时没有注意到索引范围有误，导致特征选择出错
        self.data['samples'] = df.iloc[:, 5:27].values[1:]
        self.data['labels'] = df[['NSP']].values[1:]

    def provide_split_data(self, ratio=0.6) -> tuple[dict[str, list], dict[str, list]]:
        sp_point = int(len(self.data['samples']) * ratio)

        training_samples = self.data['samples'][:sp_point]
        training_labels = self.data['labels'][:sp_point]
        testing_samples = self.data['samples'][sp_point:]
        testing_labels = self.data['labels'][sp_point:]

        return {'samples': training_samples, 'labels': training_labels}, {'samples': testing_samples, 'labels': testing_labels}
