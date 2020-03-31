
from Generic.fitting import Fit, lorentzian, area_gaussian, gaussian
from Generic.filedialogs import BatchProcess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.ndimage.filters import gaussian_filter
import os
from Generic.fitting import Fit


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


def hist_plot(variable, numbins=101, marker='r-',fig_num=2, range=None, xlabel='', ylabel='', title=''):
    freq, binedges = np.histogram(variable, bins=numbins, range=range)
    bins = 0.5 * (binedges[1:] + binedges[:-1])
    freq = freq / np.sum(freq)
    plt.figure(fig_num)
    plt.plot(bins, freq, marker)
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.title = title
    return bins, freq

def scatter_plot(x,y,marker='rx',fig_num=1, xlabel='',ylabel='', title=''):
    plt.figure(fig_num)
    plt.plot(x,y,marker)
    plt.xlabel=xlabel
    plt.ylabel=ylabel
    plt.title=title


def fit_data(x,y):
    f=Fit('linear',x=x,y=y)
    f.add_params([-1,0],lower=[None,None],upper=[None,None])
    f.fit()
    #f.plot_fit()
    #plt.show()
    return f.fit_params[0]

if __name__ == '__main__':
    file_iterator = BatchProcess(pathfilter='/media/ppzmis/data/BouncingBall_Data/NewBouncingBall/P80new/traj*f1*g3.25*.dat')
    output_path = '/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/'
    output_file = 'trajP80_3.25_f1_density.txt'

    x_normal = np.array([])



    for filename in file_iterator:#[filename, filename2, filename3]
        print(filename)
        data = np.loadtxt(filename)
        # Complete arrays
        x = np.array(data[:, 1])
        x_normal = np.append(x_normal, x)


    bins, freq = hist_plot(x_normal)

    plt.figure()
    plt.plot(bins, freq,'rx')
    plt.show()

    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/'+output_file,np.c_[bins, freq])
