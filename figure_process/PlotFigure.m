function PlotFigure(basepath,outputpath,isVisible)

% 定义所有要读取的文件路径模式
filePattern = strcat(basepath,'*.evaluation.*.csv');

% 获取所有符合模式的文件名
evaluationFiles = dir(fullfile(filePattern));

% 初始化变量名以匹配数据列
variableNamesSummary = {'Classifier', 'TrainingSetSize', 'ValidationSetSize', 'TestSetSize', 'KValueRange', 'OptimalK', 'WorstK', 'MedianK', 'OptimalAUROC', 'WorstAUROC', 'MedianAUROC'};
variableNamesEvaluation = {'K', 'TrueLabel', 'PredictedLabel', 'PredProbClass1', 'PredProbClass2', 'PredProbClass3'};

% 初始化结构体数组来保存每一对summary和evaluation文件的信息
fileInfo = struct('Classifier', {}, 'SummaryFile', {}, 'EvaluationFile', {}, 'MedianK', {}, 'Round', {});

% 读取每个summary文件并提取中位数K值，同时存储文件对信息
for fileIdx = 1:length(evaluationFiles)
    evaluationFile = evaluationFiles(fileIdx).name;
    % 使用正则表达式提取文件名中的轮次数
    roundMatch = regexp(evaluationFile, 'evaluation\.(\d+)\.', 'tokens');
    if isempty(roundMatch) || isempty(roundMatch{1})
        warning('Could not extract round number from file name: %s', evaluationFile);
        continue;
    end
    roundNumber = str2double(roundMatch{1}{1}); % 提取并转换为数字

    % 构建summary文件路径 (假设evaluation和summary文件在同一目录下，且只有文件名不同)
    summaryFile = strrep(evaluationFile, '.evaluation.', '.summary.');

    % 确保summary文件存在
    if exist(fullfile(basepath, summaryFile), 'file') ~= 2
        warning('Summary file not found for evaluation file: %s', evaluationFile);
        continue;
    end
    
    % 读取CSV文件中的数据
    summaryData = readtable(fullfile(basepath, summaryFile));
    
    % 设置summary文件的变量名称
    summaryData.Properties.VariableNames = variableNamesSummary;
    
    % 提取分类器类型和中位数K值
    classifierType = extractBetween(evaluationFile, ']', '.evaluation.'); % 假设文件命名规则包含分类器类型
    medianK = summaryData.MedianK(1); % 使用圆括号
    
    % 存储文件对及其信息，包括新的Round字段
    fileInfo(end+1) = struct('Classifier', string(classifierType), 'SummaryFile', summaryFile, 'EvaluationFile', evaluationFile, 'MedianK', medianK, 'Round', roundNumber);
end

% 检查是否收集到了任何有效文件信息
if isempty(fileInfo)
    error('No valid evaluation-summary file pairs found.');
end








% 提取分类器类型和轮次到单独的元胞数组
classifiers = [fileInfo.Classifier]; % 确保是字符串数组
rounds = [fileInfo.Round];

% 获取唯一的分类器类型和轮次组合
uniqueClassifiers = unique(classifiers);
for clfIdx = 1:length(uniqueClassifiers)
    classifierType = uniqueClassifiers(clfIdx);
    
    % 获取当前分类器类型下的所有轮次
    clfRounds = [fileInfo(strcmp([fileInfo.Classifier], classifierType)).Round];
    uniqueRounds = unique(clfRounds);
    
    % 对于每个唯一的分类器类型和轮次组合，创建一个新的图形窗口
    for roundIdx = 1:length(uniqueRounds)
        roundNumber = uniqueRounds(roundIdx);
        
        aspectRatio = 16 / 9; widthFig = 800; heightFig = widthFig / aspectRatio;

        fig = figure('Name', sprintf('%s, Round = %.0f', classifierType, roundNumber), 'NumberTitle', 'off','Visible', lower(logical(isVisible)),'Position', [100, 100, widthFig, heightFig]);
        
        % 绘制ROC曲线
        subplot(1, 2, 1);
        hold on;
        title(sprintf('ROC Curves for %s, Round = %.0f', classifierType, roundNumber));
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        
        % 绘制PR曲线
        subplot(1, 2, 2);
        hold on;
        title(sprintf('PR Curves for %s, Round = %.0f', classifierType, roundNumber));
        xlabel('Recall');
        ylabel('Precision');
        
        % 初始化图例字符串
        legendStrRoc = {};
        legendStrPr = {};

        % 遍历所有文件对，仅绘制与当前分类器类型和轮次匹配的结果
        relevantFiles = fileInfo(strcmp([fileInfo.Classifier], classifierType) & [fileInfo.Round] == roundNumber);
        for fileInfoIdx = 1:length(relevantFiles)
            evaluationData = readtable(fullfile(basepath, relevantFiles(fileInfoIdx).EvaluationFile));
            
            % 设置evaluation文件的变量名称
            evaluationData.Properties.VariableNames = variableNamesEvaluation;
            
            % 提取真实标签和预测分数
            trueLabels = evaluationData.TrueLabel;
            predictedScores = table2array(evaluationData(:, {'PredProbClass1', 'PredProbClass2', 'PredProbClass3'}));

            % 获取所有唯一的类别
            allUniqueClasses = unique(trueLabels);
            numClasses = numel(allUniqueClasses);

            % 将真实标签转换为独热编码
            oneHotTrueLabels = false(height(evaluationData), numClasses);
            for i = 1:height(evaluationData)
                [~, classIdx] = ismember(trueLabels(i), allUniqueClasses);
                if ~isempty(classIdx)
                    oneHotTrueLabels(i, classIdx) = true;
                end
            end
            
            % 绘制每个类别的ROC和PR曲线，并标注K值
            for classIdx = 1:numClasses
                if sum(oneHotTrueLabels(:,classIdx)) > 0 && sum(~oneHotTrueLabels(:,classIdx)) > 0
                    % 绘制ROC曲线
                    subplot(1, 2, 1);
                    [fpr, tpr, ~, aucRoc] = perfcurve(double(oneHotTrueLabels(:,classIdx)), predictedScores(:,classIdx), 1);
                    plot(fpr, tpr, '-','LineWidth', 2);
                    
                    % 更新ROC图例
                    legendStrRoc{end+1} = sprintf('Class %d, K=%.0f (AUC: %.2f)', allUniqueClasses(classIdx), relevantFiles(fileInfoIdx).MedianK, aucRoc);

                    % 绘制PR曲线
                    subplot(1, 2, 2);
                    [precision, recall, ~, aucPr] = perfcurve(double(oneHotTrueLabels(:,classIdx)), predictedScores(:,classIdx), 1, ...
                                                             'XCrit', 'reca', 'YCrit', 'prec');
                    plot(recall, precision, '-','LineWidth', 2);
                    
                    % 更新PR图例
                    legendStrPr{end+1} = sprintf('Class %d, K=%.0f (AUC: %.2f)', allUniqueClasses(classIdx), relevantFiles(fileInfoIdx).MedianK, aucPr);
                else
                    warning('Not enough positive and negative examples for Class %d with Classifier %s and Round = %.0f', allUniqueClasses(classIdx), classifierType, roundNumber);
                end
            end
        end
        
        % 更新图例
        subplot(1, 2, 1);
        legend(legendStrRoc, 'Location', 'best');
        hold off;
        
        subplot(1, 2, 2);
        legend(legendStrPr, 'Location', 'best');
        hold off;


         % 检查并创建保存目录
        saveDir = strcat(basepath,outputpath);
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end

         % 在这里插入保存SVG文件的代码
        saveFileName = sprintf('%s_Round%.0f.svg', classifierType, roundNumber);
        saveFilePath = fullfile(saveDir, saveFileName); % 修改为你想要保存的位置
        
        % 保存当前图形窗口为SVG文件
        try
            saveas(fig, saveFilePath);
            disp(['Saved ', saveFilePath]);
        catch ME
            warning('Failed to save the SVG file');
        end
    end
end