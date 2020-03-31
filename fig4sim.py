
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


def hist_plot(variable, numbins=100, marker='r-',fig_num=2, range=None, xlabel='', ylabel='', title=''):
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
    file_iterator = BatchProcess(pathfilter='/media/ppzmis/data/BouncingBall_Data/NewBouncingBall/P80new/imp*f1*g3.25*.dat')
    output_path = '/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/'
    output_file = 'P80_3.25'

    x_normal = np.array([])
    x_aftwall = np.array([])
    dvx_normal = np.array([])
    dvx_aftwall = np.array([])
    vx_normal = np.array([])
    vx_aftwall = np.array([])

    t_normal_bounce = np.array([])
    t_aftwall_bounce = np.array([])





    for filename in file_iterator:#[filename, filename2, filename3]
        print(filename)
        data = np.loadtxt(filename)
        # Complete arrays
        t = np.array(data[:, 0])
        x = np.array(data[:, 1])
        vx = np.array(data[:, 2])

        #check whether vx[0] is velocity b4 bounce or after.0.00974381480255445
        b4bounce = np.argwhere((data[:,4]==0) & (data[:,5]==1))
        b4wall = np.argwhere((data[:,4]==0) & (data[:, 5] == 2))
        aftbounce = np.argwhere((data[:, 4] == 1) & (data[:, 5] == 0))
        aftwall = np.argwhere((data[:, 4] == 2) & (data[:, 5] == 0))
        b4wallbounce = np.argwhere((data[:,4]==1) & (data[:,5]==2))
        aftwallbounce = np.argwhere((data[:, 4] == 2) & (data[:, 5] == 1))

        t_normal_bounce = np.append(t_normal_bounce, t[b4bounce])

    print(np.shape(t_normal_bounce))
    diff_t = np.append(t_normal_bounce[1:]-t_normal_bounce[:-1],np.array([0]))
    t_normal_bounce = t_normal_bounce[diff_t > 0.005]

    # filter on t_bounce
    print(t_normal_bounce[1:10])
    print(np.floor(t_normal_bounce[1:10]/(1/50)))
    phase = 2*np.pi*(t_normal_bounce - np.floor(t_normal_bounce/(1/50))*(1/50))/(1/50)
    bins, freq = hist_plot(phase)
    plt.show()
    print(bins)
    print(freq)
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_phase.txt',np.c_[bins, freq])
