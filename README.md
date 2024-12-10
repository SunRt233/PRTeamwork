# PRTeamwork 项目文档

## 项目概述
该项目是一个基于Python的机器学习项目，主要实现了一个分类器系统，支持PCA和LDA降维方法结合KNN分类器。项目包括数据读取、数据预处理、模型训练、评估结果保存和可视化等功能。

## 项目依赖
- Python 3.10
- numpy
- pandas
- sklearn
- matlab.engine 23.2.1
- prettytable
- openpyxl

## 基本思路
采集数据并进行预处理，然后使用PCA和LDA进行降维，最后使用KNN分类器进行分类。

由于KNN分类器需要调超参数，且在实现过程中发现数据中存在问题（存在离群点，类别样本不均衡），故采用一种类似交叉验证的策略确定最佳超参数。

先对整个数据集进行随机打乱，然后按照传入的数据集分割比例分割为训练集、验证集和测试集，
并且在分割过程中保证每种类别中训练集、验证集和测试集的占比也符合传入的比例，
借此减轻随机打乱带来的极端分布（比如训练集中只有一种类型的样本！）

![分割演示.svg](%E5%88%86%E5%89%B2%E6%BC%94%E7%A4%BA.svg)
## 目录结构

```
.
├── 原始数据
│        ├── 数据说明.docx
│        └── S3-CTG.xlsx
├── best_tests
│        └── pr_roc
├── figure_process
│        ├── plotAUROCFiles.m
│        └── PlotFigure.m
├── saves
│        ├── k_auroc
│        └── pr_roc
├── classifier.py
├── data.py
├── figure.py
├── main.py
├── README.md
├── requirements.list
└── save.py
```

## 文件说明

### main.py
- **功能**: 项目的主入口文件，负责启动程序并根据用户选择的模式（训练或展示）调用相应的函数。
- **关键函数**:
  - `mode_train(interactive: bool)`: 训练模式，允许用户选择降维方法、设置超参数，并进行多次训练。
  - `mode_show(interactive: bool)`: 展示模式，加载历史训练数据并进行测试。
  - `main(args)`: 主函数，解析命令行参数并启动程序。

### figure.py
- **功能**: 负责调用MATLAB引擎绘制训练结果的PR曲线和AU-ROC曲线。
- **关键函数**:
  - `draw_train_pr_roc_plots(is_interactive=False)`: 绘制训练过程中的PR曲线和AU-ROC曲线。
  - `draw_train_k_auroc_plots(is_interactive=False)`: 绘制不同K值下的AU-ROC曲线。
  - `draw_best_train_pr_roc_plots(is_interactive=False)`: 绘制最佳训练结果的PR曲线和AU-ROC曲线。

### classifier.py
- **功能**: 实现了PCA和LDA降维方法结合KNN分类器的具体实现。
- **关键类**:
  - `BaseClassifier`: 抽象基类，定义了分类器的基本接口。
  - `PCAKNNClassifier`: PCA降维后KNN分类器。
  - `LDANKNNClassifier`: LDA降维后KNN分类器。
  - `Memory`: 存储处理后的数据及其处理器和维度信息。

### save.py
- **功能**: 负责保存和加载训练结果。
- **关键类**:
  - `TrainSave`: 保存训练数据、验证数据、测试数据及评估结果。
  - `TrainSaveLoader`: 从文件中加载训练结果。

### data.py
- **功能**: 负责数据的读取和预处理。
- **关键类**:
  - `DataContainer`: 存储样本数据和对应的标签。
  - `DataProvider`: 提供数据读取和分割功能。

### figure_process\
- **功能**: 包含MATLAB脚本，用于绘制训练结果的图表。
- **关键文件**:
  - `plotAUROCFiles.m`: 读取包含AU-ROC数据的CSV文件并绘制AU-ROC曲线。
  - `PlotFigure.m`: 读取评估结果文件并绘制ROC曲线和PR曲线。

## 依赖关系
- **Python库**:
  - `argparse`: 用于解析命令行参数。
  - `os`: 用于文件路径操作。
  - `re`: 用于正则表达式匹配。
  - `datetime`: 用于时间戳生成。
  - `prettytable`: 用于生成表格。
  - `numpy` 和 `pandas`: 用于数据处理。
  - `sklearn`: 用于机器学习算法。
  - `matlab.engine`: 用于调用MATLAB引擎。

- **MATLAB脚本**:
  - `plotAUROCFiles.m` 和 `PlotFigure.m` 用于绘制训练结果的图表。

## 运行流程
1. **启动程序**:
   - 用户通过命令行启动程序，可以选择是否进入交互模式。
   - `main.py` 解析命令行参数并调用 `main` 函数。
   - `--isPresenting` 选项控制是否进入交互模式。

2. **选择模式**:
   - 用户选择运行模式（训练或展示）。
   - 根据选择的模式调用 `mode_train` 或 `mode_show` 函数。

3. **训练模式**:
   - 用户选择降维方法（PCA或LDA）和设置超参数。
   - 系统进行多次训练并保存结果。

4. **展示模式**:
   - 系统列出历史训练结果。
   - 用户选择要展示的历史结果，系统加载并进行测试。
   - 绘制并保存测试结果的图表。

5. **查看生成的figure**:
   - 用户可以查看生成的图表，包括PR曲线和AU-ROC曲线。
   - 迭代过程的图表保存在 `saves/pr_roc`、`saves/k_auroc` 目录下。
   - 最佳训练结果的图表保存在 `best_tests/pr_roc` 目录下。

## 总结
该项目结构清晰，功能明确，涵盖了数据读取、预处理、模型训练、评估结果保存和可视化的完整流程。通过MATLAB脚本的引入，增强了图表绘制的灵活性和美观度。
