import argparse
import os
import re
from datetime import datetime
from prettytable import PrettyTable
from classifier import PCAKNNClassifier, LDAKNNClassifier, BaseClassifier
from data import DataProvider
from save import TrainSave, TrainSaveLoader
from figure import draw_best_train_pr_roc_plots, draw_train_pr_roc_plots, draw_train_k_auroc_plots
from concurrent.futures import ThreadPoolExecutor
# 一些常量
saves_path = r'./saves'
best_test_saves_path = r'./best_tests'
default_k_min = 5
default_k_max = 25
default_shuffle_turns = 9

running_modes = [
    {'编号': 1, '模式': '训练', '描述': '批量运行调整超参数'},
    {'编号': 2, '模式': '展示', '描述': '加载历史数据并测试'}
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
    k_min = default_k_min
    k_max = default_k_max
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

    shuffle_turn = default_shuffle_turns
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

    timelabel = datetime.now().strftime("[%Y-%m-%d_%H_%M_%S]")

    def task(classifier,t,i):
        print(f'开始训练 {classifier.__class__.__name__}，第 {t} 次打乱')
        train_save = train(classifier, data_provider, k_min, k_max, shuffle=True)
        train_save.save(saves_path, f'{timelabel}{train_save.classifier}', t)
        return train_save

    def callback(future):
        train_saves.append(future.result())

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            i = 1
            total_tasks = shuffle_turn * len(classifiers)
            current_completed_tasks = -1

            for t in range(1, shuffle_turn + 1):
                for classifier in classifiers:
                    future = executor.submit(task, classifier, t,i)
                    future.add_done_callback(callback)
                    futures.append(future)
                    i += 1

            import time
            while current_completed_tasks != total_tasks:
                l = len(train_saves)
                # if l > current_completed_tasks:
                    # 停2s
                time.sleep(4)
                current_completed_tasks = l
                print(f'进度：{current_completed_tasks}/{total_tasks}')

    except KeyboardInterrupt:
        executor.shutdown(cancel_futures=True)

    print('训练完成')
    # Table打印summaries
    table = PrettyTable()
    table.field_names = ['分类器', '训练集大小', '验证集大小', '测试集大小', 'K值范围', '最优K值', '最差K值',
                         '中位数K值', '最优au_roc', '最差au_roc', '中位数au_roc']
    for train_save in train_saves:
        summary = train_save.summary()
        table.add_row([round(item, 4) if isinstance(item, float) else item for item in summary.values()])
    print(table)

    print('绘制并保存 K-AU-ROC 曲线')
    draw_train_pr_roc_plots(is_interactive=False)
    print('绘制并保存 PR 曲线和 AU-ROC 曲线')
    draw_train_k_auroc_plots(is_interactive=interactive)


def mode_show(interactive: bool):
    def list_saves():
        """
        列出saves下的TrainSave文件名
        :return: saves下的TrainSave文件名
        """
        table = PrettyTable()
        table.field_names = ['编号', '时间', '分类器', '最大打乱轮数']

        # 定义正则表达式
        pattern = re.compile(r'^\[(?P<time>[^\]]+)\](?P<classifier>[^.]+)\.summary(?:\.(?P<shuffle>\d+))?\.csv$')

        # 使用字典存储每个时间-分类器组合的最大打乱轮数
        max_shuffle_dict = {}

        for file in os.listdir(saves_path):
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
        return max_shuffle_dict

    def load_saves(saves_dict):
        """
        加载 TrainSaves，为每个分类器选出最佳 k 值
        选取原则：每个分类器得到n个随机打乱的数据集，相当于每个分类器运行n轮，
        每轮运行都会尝试范围内的全部 k 值并计算 au_roc，
        选出 au_roc 的中位数和对应 k 值作为本轮的结果，
        接着在产生的 n 个 au_roc 中选出最大的 au_roc 对应的 k 值作为该分类器的最佳 k 值。
        :param saves_dict:
        :return: train_saves 和 best_saves
        """
        loader = TrainSaveLoader()
        train_saves: dict[str, dict[int, TrainSave]] = {}
        best_saves: dict[str, tuple[TrainSave, int, float]] = {}

        # 逐分类器加载TrainSave
        for (time, classifier), max_shuffle in saves_dict.items():
            if classifier not in train_saves.keys():
                train_saves[classifier] = {}

            # 每轮打乱中根据 au_roc 的中位数选出一个 k 值，
            # 然后选出对应 au_roc 最大的 k 作为最佳 k 值
            best_k = -1
            best_au_roc = -1
            best_save = None

            for t in range(1, max_shuffle + 1):
                # 加载每轮打乱的TrainSave
                print(f'加载 {classifier} 第 {t} 轮打乱的训练数据')
                train_save = loader.load(path=saves_path, name=f'[{time}]{classifier}', shuffle_turns=t)
                summary = train_save.summary()
                median_k = summary['中位数K值']
                median_au_roc = summary['中位数au_roc']

                # 一个简单的选择排序
                if median_au_roc > best_au_roc:
                    best_k = median_k
                    best_au_roc = median_au_roc
                    # 更新最佳 TrainSave
                    best_save = train_save
                # 保存全部 TrainSave
                train_saves[classifier][t] = train_save

            best_saves[classifier] = (best_save, best_k, best_au_roc)

        # Table 打印每个分类器的最佳超参数 k 值，以及对应的 au_roc
        table = PrettyTable()
        table.field_names = ['分类器', '最佳K值', '对应au_roc']
        for classifier, (train_save, k, au_roc) in best_saves.items():
            table.add_row([classifier, k, au_roc])
        print(table)

        return train_saves, best_saves

    def test(classifier: BaseClassifier, train_save: TrainSave, k: int):
        print(f'开始使用 {classifier_name} 预测测试集')
        # 使用记录的训练集重新训练模型
        classifier.train(train_save.train_data, k)
        # 使用记录的测试集进行预测
        predict_result, predict_probability = classifier.predict(train_save.test_data.samples)
        # 结果保存为新的 TrainSave
        new_train_save = TrainSave(classifier_name, train_save.train_data, train_save.verification_data,
                                   train_save.test_data)
        new_train_save.add_evaluation_result(k, predict_result, predict_probability,
                                             true_labels=train_save.test_data.labels)
        return new_train_save

    saves_dict = list_saves()
    _, best_saves = load_saves(saves_dict)

    # 根据最优 TrainSave，使用其记录的训练集重新训练模型，使用其记录的测试集进行预测
    tests = []
    for classifier_name, (train_save, k, au_roc) in best_saves.items():
        match classifier_name:
            case PCAKNNClassifier.__name__:
                tests.append(test(PCAKNNClassifier(), train_save, k))
            case LDAKNNClassifier.__name__:
                tests.append(test(LDAKNNClassifier(), train_save, k))
            case _:
                raise ValueError(f'{classifier_name} 不支持')

    # 保存测试结果
    os.makedirs(best_test_saves_path, exist_ok=True)
    time = datetime.now().strftime("[%Y-%m-%d_%H_%M_%S]")
    for test in tests:
        test.save(best_test_saves_path, f'{time}{test.classifier}', 0)

    print('绘制并保存 PR 曲线和 AU-ROC 曲线')
    draw_best_train_pr_roc_plots(is_interactive=interactive)

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

    os.makedirs(saves_path, exist_ok=True)
    should_exiting = False
    if args.isPresenting:
        print("当前为演示模式，启用交互")
    while args.isPresenting and (not should_exiting):
        # 打印表格
        print(table)
        print("可输入 exit 结束运行")
        mode = input(prompt('选择运行模式', suffix='：'))
        match mode:
            case '1':
                mode_train(interactive=True)
                # 回车返回上一级
                input('回车返回上一级')
                # break
            case '2':
                mode_show(interactive=True)
                # 回车返回上一级
                input('回车返回上一级')
            case 'exit':
                should_exiting = True
            case _:
                print('请输入正确的模式编号！')
    if not args.isPresenting:
        print('当前非演示模式，禁用交互，全部按默认参数运行')
        mode_train(interactive=False)
        mode_show(interactive=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isPresenting', action='store_true', help='是否展示运行模式', default=False)
    args = parser.parse_args()
    main(args)
