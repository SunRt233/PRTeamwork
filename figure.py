import matlab.engine

def draw_train_plots():
    # 使用matlab engine 调用 figure_process/PlotFigure.m
    eng = matlab.engine.start_matlab()

    # 切换工作目录
    eng.cd(r'./figure_process',nargout=0)
    eng.quit()