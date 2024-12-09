import pandas as pd
from sklearn.metrics import roc_auc_score

from classifier import BaseClassifier
from data import DataContainer


class TrainSave:
    summary_keys = ['分类器', '训练集大小', '验证集大小', '测试集大小', '最优K值','验证集AU-ROC']

    def __init__(self, classifier: str, train_data: DataContainer, verification_data: DataContainer,
                 test_data: DataContainer):
        self.classifier = classifier
        self.train_data = train_data
        self.verification_data = verification_data
        self.test_data = test_data
        self.evaluation_results = {}
        self.au_roc : dict[int, float] = {}

        col = [f'特征{i}' for i in range(self.train_data.num_features())]
        col.extend(['标签'])

        tr_d = [[*self.train_data.samples[i], self.train_data.labels[i]] for i in range(self.train_data.num_samples())]
        tr_df = pd.DataFrame(columns=col, data=tr_d)
        ve_d = [[*self.verification_data.samples[i], self.verification_data.labels[i]] for i in
                range(self.verification_data.num_samples())]
        ve_df = pd.DataFrame(columns=col, data=ve_d)
        te_d = [[*self.test_data.samples[i], self.test_data.labels[i]] for i in range(self.test_data.num_samples())]
        te_df = pd.DataFrame(columns=col, data=te_d)

        self.dataset_dfs = {'train': tr_df, 'verification': ve_df, 'test': te_df}


    def add_evaluation_result(self, k, predict_result, predict_probability):
        au_roc = roc_auc_score(y_true=self.verification_data.labels, y_score=predict_probability, multi_class='ovo',average='weighted',labels=self.verification_data.classes())
        self.au_roc[k] = au_roc
        self.evaluation_results[k] = [[self.verification_data.labels[i], predict_result[i], *predict_probability[i]] for
                                      i in range(self.verification_data.num_samples())]

    def save(self, path, name, shuffle_turns = None | int):
        suffix = 'csv'
        if shuffle_turns is not None:
            suffix = f'{shuffle_turns}.csv'

        # 用pandas保存为CSV
        # 训练集，验证集，测试集保存为三个CSV
        for key, df in self.dataset_dfs.items():
            df.to_csv(f'{path}/{name}.{key}.{suffix}', index=False)

        # 保存评估结果
        data_to_save = []
        for k in self.evaluation_results.keys():
            for result in self.evaluation_results[k]:
                data_to_save.append([k,*result])
        col = ['K','真实标签','预测标签',*[f'预测为{i+1}的概率' for i in range(self.train_data.num_classes())]]
        pd.DataFrame(data_to_save, columns=col).to_csv(f'{path}/{name}.evaluation.{suffix}', index=False)

        au_roc_data = []
        for k in self.au_roc.keys():
            au_roc_data.append([k,self.au_roc[k]])
        col = ['K','验证集AU-ROC']
        pd.DataFrame(au_roc_data, columns=col).to_csv(f'{path}/{name}.au_roc.{suffix}', index=False)

        # 保存summary()返回的字典到csv
        pd.DataFrame([self.summary()]).to_csv(f'{path}/{name}.summary.{suffix}',index=False)

    def summary(self) -> dict:
        # 返回一个字典
        return {
            '分类器': self.classifier,
            '训练集大小': self.train_data.num_samples(),
            '验证集大小': self.verification_data.num_samples(),
            '测试集大小': self.test_data.num_samples(),
            'K值范围': f'{min(self.evaluation_results.keys())}~{max(self.evaluation_results.keys())}',
            # au_roc的键
            '最优K值': max(self.au_roc, key=self.au_roc.get),
            '最差K值': min(self.au_roc, key=self.au_roc.get),
            '中位数K值': sorted(self.au_roc,key=self.au_roc.get)[len(self.au_roc) // 2],
            '最优au_roc': float(max(self.au_roc.values())),
            '最差au_roc': float(min(self.au_roc.values())),
            '中位数au_roc': float(sorted(self.au_roc.values())[len(self.au_roc) // 2])
        }
