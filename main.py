import argparse
import os
import re
from datetime import datetime

import numpy as np
from prettytable import PrettyTable

from classifier import PCAKNNClassifier, LDAKNNClassifier, BaseClassifier
from data import DataProvider
from save import TrainSave

saves = r'./saves'

running_modes = [
    {'编号': 1, '模式': '训练', '描述': '批量运行调整超参数'},
    {'编号': 2, '模式': '展示', '描述': '加载历史最优并展示'}
]

support_methods = [
    {'编号': 1, '方法': 'LDA', '描述': 'LDA降维+KNN分类'},
    {'编号': 2, '方法': 'PCA', '描述': 'PCA降维+KNN分类'}
]


def mode_train(interactive: bool):
    def train(classifier: BaseClassifier, data_provider: DataProvider, k_min, k_max, shuffle=False):
        train_data, verification_data, test_data = (
            data_provider.provide_balanced_split_data(train_ratio=0.6,
                                                      verification_ratio=0.2,
                                                      shuffle=shuffle))
        save = TrainSave(classifier.__class__.__name__, train_data, verification_data, test_data)
        for k in range(k_min, k_max + 1):
            classifier.train(train_data, k=k)
            result, probability = classifier.predict(verification_data.samples)
            save.add_evaluation_result(k, result, probability)
        return save

    data_provider = DataProvider(r'./原始数据/S3-CTG.xlsx')
    data_provider.load_data()

    classifiers: list[BaseClassifier] = []

    table = PrettyTable()
    table.field_names = ["编号", "方法", "描述"]
    for item in support_methods:
        table.add_row([item['编号'], item['方法'], item['描述']])

    while interactive:
        print(table)
        match input(prompt('选择要使用的方法（输入all表示使用所有方法）', suffix='：')):
            case '1':
                classifiers.append(LDAKNNClassifier())
                break
            case '2':
                classifiers.append(PCAKNNClassifier())
                break
            case 'all':
                break
            case _:
                print('请输入正确的方法编号！')
                continue

    if not interactive or len(classifiers) == 0:
        print(table)
        print('已选择所有方法')
        classifiers.append(PCAKNNClassifier())
        classifiers.append(LDAKNNClassifier())

    # 输入超参数 K 的搜索范围，默认最小值为5，默认最大值为25
    k_min = 5
    k_max = 25
    while interactive:
        input_str = input(prompt(f'输入 K 的最小值，默认为 {k_min}', suffix='：'))
        match input_str:
            case '':
                break
            case _:
                try:
                    k_min = int(input_str)
                    if k_min < 5:
                        print('K 的最小值不能小于5！')
                        continue
                    break
                except ValueError:
                    print('请输入正确的整数！')
    while interactive:
        input_str = input(prompt(f'输入 K 的最大值，默认为 {k_max}', suffix='：'))
        match input_str:
            case '':
                break
            case _:
                try:
                    k_max = int(input_str)
                    if k_max < k_min:
                        print('K 的最大值不能小于K的最小值！')
                        continue
                    break
                except ValueError:
                    print('请输入正确的整数！')

    if not interactive:
        # 打印K默认范围
        print(f'K 的默认范围是 {k_min} ~ {k_max}')

    shuffle_turn = 5
    while interactive:
        input_str = input(prompt(f'输入打乱次数，默认为 {shuffle_turn}', suffix='：'))
        match input_str:
            case '':
                break
            case _:
                try:
                    shuffle_turn = int(input_str)
                    if shuffle_turn < 0:
                        print('打乱次数不能小于0！')
                        continue
                    break
                except ValueError:
                    print('请输入正确的整数！')
    if not interactive:
        print(f'打乱次数默认为 {shuffle_turn}')

    train_saves = []
    print('开始训练')

    time = datetime.now().strftime("[%Y-%m-%d_%H_%M_%S]")
    for t in range(1, shuffle_turn + 1):
        for classifier in classifiers:
            print(f'开始训练 {classifier.__class__.__name__}，第 {t} 次打乱')
            train_save = train(classifier, data_provider, k_min, k_max, shuffle=True)
            train_save.save(saves, f'{time}{classifier.__class__.__name__}', t)
            train_saves.append(train_save)

    print('训练完成')
    # Table打印summaries
    table = PrettyTable()
    table.field_names = ['分类器', '训练集大小', '验证集大小', '测试集大小', 'K值范围', '最优K值', '最差K值',
                         '中位数K值', '最优au_roc', '最差au_roc', '中位数au_roc']
    for train_save in train_saves:
        summary = train_save.summary()
        table.add_row([round(item,4) if isinstance(item,float) else item for item in summary.values()])
    print(table)


def mode_show():
    # Table列出saves下的TrainSave文件
    table = PrettyTable()
    table.field_names = ['编号', '时间', '分类器', '最大打乱轮数']

    # 定义正则表达式
    pattern = re.compile(r'^\[(?P<time>[^\]]+)\](?P<classifier>[^.]+)\.summary(?:\.(?P<shuffle>\d+))?\.csv$')

    # 使用字典存储每个时间-分类器组合的最大打乱轮数
    max_shuffle_dict = {}

    for file in os.listdir(saves):
        match = pattern.match(file)
        if match:
            # 提取时间
            time = match.group('time')
            # 提取分类器
            classifier = match.group('classifier')
            # 提取打乱次数（如果有）
            shuffle = int(match.group('shuffle')) if match.group('shuffle') else 0

            # 更新最大打乱轮数
            key = (time, classifier)
            if key in max_shuffle_dict:
                max_shuffle_dict[key] = max(max_shuffle_dict[key], shuffle)
            else:
                max_shuffle_dict[key] = shuffle

    # 将字典中的信息添加到表格中
    i = 1
    for (time, classifier), max_shuffle in max_shuffle_dict.items():
        table.add_row([i, time, classifier, max_shuffle if max_shuffle > 0 else '无'])
        i += 1

    print(table)


def prompt(str, prefix='', suffix=' >'):
    if len(prefix) > 0:
        prefix = f'({prefix}) '
    return f'{prefix}{str}{suffix} '


def main(args):
    # 选择运行模式，要验证输入是否合法
    table = PrettyTable()
    table.field_names = ["编号", "模式", "描述"]

    # 添加行数据
    for item in running_modes:
        table.add_row([item['编号'], item['模式'], item['描述']])

    os.makedirs(saves, exist_ok=True)

    while args.isPresenting:
        # 打印表格
        print(table)
        mode = input(prompt('选择运行模式', suffix='：'))
        match mode:
            case '1':
                mode_train(interactive=True)
                break
            case '2':
                mode_show()
                break
            case _:
                print('请输入正确的模式编号！')
    if not args.isPresenting:
        print('当前非演示模式')
        mode_train(interactive=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isPresenting', action='store_true', help='是否展示运行模式', default=False)
    args = parser.parse_args()
    main(args)
