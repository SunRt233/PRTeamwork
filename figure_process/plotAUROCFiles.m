function plotAUROCFiles(basepath, outputpath, isVisible)
    % PLOTAUROCFILES 读取包含 'au_roc' 字样的CSV文件并绘制AUROC曲线。
    %
    % 输入:
    %   basepath - 文件所在的目录路径
    %   outputpath - 图形输出保存的目录路径
    %   isVisible - 逻辑值（true 或 false），用于控制图形窗口的可见性

    % 使用 dir 函数查找包含 'au_roc' 字样的文件
    files = dir(fullfile(basepath, '*au_roc*.csv'));

    % 检查是否有匹配的文件
    if isempty(files)
        error('No files with "au_roc" in the name found in the specified directory.');
    end

    % 初始化变量名
    variableNamesAu_roc = {'k', 'au_roc'};
    
    % 创建一个容器来保存每个分类器的图形句柄
    classifierFigs = containers.Map('KeyType','char','ValueType','any');

    % 遍历每个文件
    for i = 1:length(files)
        % 获取文件名
        fileName = fullfile(basepath, files(i).name);
        
        % 提取分类器类型和打乱次数
        [classifierType, roundNumber] = extractClassifierInfo(files(i).name);
    
        % 读取CSV文件
        data = readtable(fileName);
        data.Properties.VariableNames = variableNamesAu_roc;
       
        % 提取k和au_roc列
        k = data.k;
        au_roc = data.au_roc;

        % 如果该分类器还没有对应的图形，则创建新的图形窗口
        if ~isKey(classifierFigs, classifierType)
            fig = figure('Name', classifierType, 'NumberTitle', 'off', 'Visible', logical(isVisible));
            hold on;
            classifierFigs(classifierType) = fig;  % 将图形句柄存入Map中
        else
            fig = classifierFigs(classifierType);  % 获取已存在的图形句柄
            set(0, 'CurrentFigure', fig)
            set(fig, 'Visible', logical(isVisible)); 
        end
        
        % 绘制曲线
        h = plot(k, au_roc, '-o', 'LineWidth', 1, 'MarkerFaceColor', 'b','MarkerSize', 4);
        set(h, 'DisplayName', sprintf('Round %.0f', roundNumber));

        % 设置图形标题和轴标签
        title(sprintf('AUROC vs. k for %s', classifierType));
        xlabel('k');
        ylabel('AUROC');

        % 添加图例
        legend show;
        
        % 保存图像到指定路径
        saveDir = fullfile(basepath, outputpath);
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
        
        % 仅当所有轮次都添加完毕后保存图像
        if i == length(files) || ~strcmp(classifierType, extractClassifierInfo(files(min(i+1, length(files))).name))
            saveFileName = sprintf('%s.svg', classifierType);
            saveFilePath = fullfile(saveDir, saveFileName); % 修改为你想要保存的位置
            
            try
                saveas(fig, saveFilePath);
                disp(['Saved ', saveFilePath]);
            catch ME
                warning('Failed to save the SVG file');
            end
        end
    end
    
    function [classifierType, roundNumber] = extractClassifierInfo(filename)
    % 提取分类器类型和轮次数的辅助函数
    % 假设文件名格式为 [timestamp]ClassifierType.au_roc.[number].csv
    
    % 使用正则表达式匹配文件名
    match = regexp(filename,  '^\[.*?\](LDAKNNClassifier|PCAKNNClassifier)\.au_roc\.(\d+)\.csv$', 'tokens');
    
    if ~isempty(match)
        classifierType = match{1}{1};  % 获取分类器类型 (LDAKNN 或 PCAKNN)
        roundNumber = str2double(match{1}{2});  % 获取轮次数
    else
        error(['Invalid filename format: ', filename]);
    end
end
end