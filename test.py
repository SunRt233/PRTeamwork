from data import DataProvider
from classifier import Classifier

data_provider = DataProvider(r'./数据及说明/S3-CTG.xlsx')
data_provider.read_data()
train_data, test_data = data_provider.provide_split_data(ratio=0.001)

classifier = Classifier()
classifier.train(train_data, Classifier.Method.PCA)
result = classifier.predict(test_data['samples'])

# print(result)
# print(test_data['labels'])
correct_num = 0
total_num = len(result)
for i in range(0,len(result)-1):
    matches = False
    if result[i] == test_data['labels'][i].all():
        correct_num += 1
        matches = True
    print(i,result[i],test_data['labels'][i],'匹配:',matches)
print('正确率:',correct_num/total_num)
