from data import DataProvider
from classifier import Classifier

data_provider = DataProvider(r'./数据及说明/S3-CTG.xlsx')
data_provider.read_data()
classifier = Classifier()
def run_one_turn(ratio: float,shuffle: bool):
    train_data, test_data = data_provider.provide_split_data(ratio,shuffle)
    classifier.train(train_data, Classifier.Method.PCA,dimens=10)
    result = classifier.predict(test_data.samples)
    correct_rate = match_test(result,test_data)
    return {'正确率':correct_rate,'训练集测试集分割比例':ratio,'是否打乱数据':shuffle}
def match_test(result,test_data):
    correct_num = 0
    total_num = len(result)
    for i in range(0,len(result)-1):
        matches = False
        # 这里出现过重大错误，因为test_data.labels[i]实际上是从DataFrame中返回的Series转换来的list
        # 一开始直接使用了test_data.labels[i].all()，导致部分结果总是为True，出现了“假阳性”
        # 因此，这里需要再进一步访问test_data.labels[i][0]，再进行判断
        if result[i] == test_data.labels[i][0]:
            correct_num += 1
            matches = True
        # print(i,result[i],test_data.labels[i][0],'匹配:',matches)
    # print('正确率:',correct_num/total_num)
    return correct_num/total_num

def main():
    modes = {1: '批量测试查找最优参数'}
    print(modes)
    mode = int(input('选择运行模式：'))
    match mode:
        case 1:
            ratio_step = float(input('输入数据集划分比例增长步长：'))
            shuffle = input('是否打乱数据？[Y/N]') == 'Y'
            turns = int(input('每种参数搭配的运行轮数：'))
            print('开始批量测试')
            global_results = []
            ratio = round(ratio_step,2)
            while ratio < 1:
                print('--------------------------------------------------------------------------------')
                print('当前测试比例：',ratio)
                current_results = []

                for i in range(1,turns + 1):
                    result = run_one_turn(ratio,shuffle)
                    current_results.append(result)
                    print('第',i,'轮测试结果：',result)
                # 计算平均正确率
                correct_rate = sum([result['正确率'] for result in current_results]) / len(current_results)
                # 最好的正确率
                best_correct_rate = max([result['正确率'] for result in current_results])
                # 最差的正确率
                worst_correct_rate = min([result['正确率'] for result in current_results])
                # 偏差
                deviation = best_correct_rate - worst_correct_rate
                print('平均正确率：',correct_rate,' 最佳正确率：',best_correct_rate,' 最差正确率：',worst_correct_rate,' 偏差：',deviation)
                # 保存当前ratio下平均结果到全局结果中
                global_results.append({'正确率':correct_rate,'训练集测试集分割比例':ratio,'是否打乱数据':shuffle})

                ratio += ratio_step
                ratio = round(ratio,2)
                print('--------------------------------------------------------------------------------')

            # 选出最好的结果
            best_result = max(global_results,key=lambda x:x['正确率'])
            print('全局最优结果：',best_result)


main()