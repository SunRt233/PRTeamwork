import matlab.engine

eng = matlab.engine.start_matlab()

# 切换工作目录
eng.cd(r'./figure_process',nargout=0)
eng.warning('off',nargout=0)

def draw_train_pr_roc_plots(is_interactive=False):
    # 使用matlab engine 调用 figure_process/PlotFigure.m
    eng.PlotFigure('../saves/','./pr_roc',is_interactive,nargout=0)

def draw_train_k_auroc_plots(is_interactive=False):
    eng.plotAUROCFiles('../saves/','./k_auroc',is_interactive,nargout=0)

def draw_best_train_pr_roc_plots(is_interactive=False):
    eng.PlotFigure('../best_tests/','./pr_roc',is_interactive,nargout=0)