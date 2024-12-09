import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future

import pandas as pd
import numpy as np

from classifier import Classifier
from data import DataProvider
from sklearn.metrics import roc_auc_score

from roc import plot_roc

data_provider = DataProvider(r'../数据及说明/S3-CTG.xlsx')
data_provider.load_data()


def run_once(ratio: float, shuffle: bool, method: Classifier.Method):
    classifier = Classifier()
    train_data, test_data = data_provider.provide_split_data(ratio, shuffle)
    classifier.train(train_data, method)
    result,probability = classifier.predict(test_data.samples)
    roc_ = roc(np.ravel(test_data.labels),probability)
    correct_rate = match_test(result, test_data)
    plot_roc(classifier, test_data)
    return {'正确率': correct_rate, '训练集测试集分割比例': ratio, '是否打乱数据': shuffle, '方法': method.name,'ROC':roc_}

def roc(real_labels,probabilities) -> float:
    return roc_auc_score(y_true=real_labels, y_score=probabilities, multi_class='ovo',average='weighted',labels=[1,2,3])

def match_test(result, test_data):
    correct_num = 0
    total_num = len(result)
    for i in range(0, len(result)):
        if result[i] == test_data.labels[i][0]:
            correct_num += 1
    return correct_num / total_num


def arrange_result(global_results, log_enabled=False) -> list:
    arr = []
    for ratio, results in global_results.items():
        # 计算平均正确率
        avg_correct_rate = sum([result['正确率'] for result in results]) / len(results)
        # 最好的正确率
        best_correct_rate = max([result['正确率'] for result in results])
        # 最差的正确率
        worst_correct_rate = min([result['正确率'] for result in results])
        # 偏差
        correct_rate_deviation = best_correct_rate - worst_correct_rate

        avg_roc = sum([result['ROC'] for result in results]) / len(results)
        best_roc = max([result['ROC'] for result in results])
        worst_roc = min([result['ROC'] for result in results])
        roc_deviation = best_roc - worst_roc
        if log_enabled:
            print(
                f'分割比例：{ratio}\t运行次数：{len(results)}\t平均正确率：{avg_correct_rate}\t最佳正确率：{best_correct_rate}\t最差正确率：{worst_correct_rate}\t偏差：{correct_rate_deviation}')
        # 保存当前ratio下平均结果到全局结果中
        arr.append({'平均正确率': avg_correct_rate, '训练集测试集分割比例': ratio, '运行次数': len(results), '最佳正确率': best_correct_rate, '最差正确率': worst_correct_rate, '偏差': correct_rate_deviation, '平均ROC': avg_roc, '最佳ROC': best_roc, '最差ROC': worst_roc, 'ROC偏差': roc_deviation})
    return arr


def update_global_results(ratio: float, global_results):
    def func(future: Future):
        result = future.result()
        if ratio in global_results.keys():
            global_results[ratio].append(result)
        else:
            global_results[ratio] = [result]

    return func


def main():
    modes = {1: '批量测试查找最优参数'}
    print(modes)
    mode = int(input('选择运行模式：'))
    match mode:
        case 1:
            ratio_step = float(input('输入数据集划分比例增长步长：'))
            shuffle = input('是否打乱数据？[Y/N]') == 'Y'
            turns = int(input('每种参数搭配的运行轮数：'))
            m = int(input('输入要使用的方法（1.PCA 2.LDA）：'))
            method = Classifier.Method.PCA if m == 1 else Classifier.Method.LDA
            max_threads = int(input('输入最大线程数量：'))
            print('开始批量测试')
            global_results: dict = {}

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = []
                ratio = round(ratio_step, 2)
                while 0 < ratio < 1:
                    for i in range(1, turns + 1):
                        future: Future = executor.submit(run_once, ratio, shuffle, method)
                        future.add_done_callback(update_global_results(ratio, global_results))
                        futures.append(future)

                    ratio += ratio_step
                    ratio = round(ratio, 2)

                completed_count = 0
                total_tasks = len(futures)
                for _ in concurrent.futures.as_completed(futures):
                    completed_count += 1
                    print(f'进度：{completed_count}/{total_tasks}')

            # 选出最好的结果
            arr = arrange_result(global_results, False)
            best_result = max(arr, key=lambda x: x['平均正确率'])
            print('结果总结')
            print('划分比例增量步长：', ratio_step,'是否打乱数据：', shuffle, '方法：', method.name, '每轮运行次数：', turns, '线程数量：', max_threads)
            print('全局最优结果：', best_result)

            pd.DataFrame(arr).to_csv('result.csv')


main()
