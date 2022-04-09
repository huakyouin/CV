import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from collections import namedtuple
from matplotlib import cm
from scipy import interpolate
import seaborn as sns
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def draw_ts(data,legend):
    for i in range(len(data)):
        plt.plot(data[i],label=legend[i])
    plt.xlabel('batchs')
    plt.legend()
    plt.show()
    return()



