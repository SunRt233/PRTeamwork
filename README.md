# PRTeamwork

# 项目结构

- 用pands库解析数据
- 用scikit-learn库做PCA降维+KNN学习

# xxx
- 先不降维，直接计算了主成分的累计解释方差，然后超过0.95则停止，结果是降维到13维
- 调参迭代的过程中发现，降维到2维，训练比例为0.7，准确率可以达到0.8430141287284144