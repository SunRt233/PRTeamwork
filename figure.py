import matlab
from matlab import engine

eng_future = matlab.engine.start_matlab(background=True)
print('后台异步启动MatlabPyEngine')
cded = False
def get_eng():
    global eng_future
    global cded
    eng = eng_future.result()
    if not cded:
        cded = True
        eng.cd(r'./figure_process', nargout=0)
        eng.warning('off', nargout=0)
    return eng


def draw_train_pr_roc_plots(is_interactive=False):
    # 使用matlab eng_futureine 调用 figure_process/PlotFigure.m
    get_eng().PlotFigure('../saves/', './pr_roc', is_interactive, nargout=0)

def draw_train_k_auroc_plots(is_interactive=False):
    get_eng().plotAUROCFiles('../saves/', './k_auroc', is_interactive, nargout=0)

def draw_best_train_pr_roc_plots(is_interactive=False):
    get_eng().PlotFigure('../best_tests/', './pr_roc', is_interactive, nargout=0)
