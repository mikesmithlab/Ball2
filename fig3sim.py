
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


if __name__ == '__main__':
    #Read in data
    file_iterator = BatchProcess(pathfilter='/media/ppzmis/data/BouncingBall_Data/NewBouncingBall/P80new/imp*f1*g3.25*.dat')
    output_path = '/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/'
    output_file = 'P80_3.25'

    #Prepare arrays to store data
    x_normal = np.array([])
    x_aftwall = np.array([])
    dvx_normal = np.array([])
    dvx_aftwall = np.array([])
    vx_normal = np.array([])
    vx_aftwall = np.array([])
    vrw_normal = np.array([])
    vrw_aftwall = np.array([])
    t_normal_bounce = np.array([])
    t_aftwall_bounce = np.array([])
    w_normal = np.array([])
    dw_normal = np.array([])

    for filename in file_iterator:
        print(filename)
        data = np.loadtxt(filename)
        # Complete arrays
        t = np.array(data[:, 0])
        x = np.array(data[:, 1])
        vx = np.array(data[:, 2])
        vrw = np.array(data[:, 3])

        #check whether vx[0] is velocity b4 bounce or after.0.00974381480255445
        b4bounce = np.argwhere((data[:,4]==0) & (data[:,5]==1))
        b4wall = np.argwhere((data[:,4]==0) & (data[:, 5] == 2))
        aftbounce = np.argwhere((data[:, 4] == 1) & (data[:, 5] == 0))
        aftwall = np.argwhere((data[:, 4] == 2) & (data[:, 5] == 0))
        b4wallbounce = np.argwhere((data[:,4]==1) & (data[:,5]==2))
        aftwallbounce = np.argwhere((data[:, 4] == 2) & (data[:, 5] == 1))

        #normal bounces, These should be (0,1),(1,0),(0,1)
        # The returned indices should be inserted into the complete arrays.
        #The index refers to the number of the before bounce.
        #To get after bounce + 1. B4 bounce -1
        temp = np.setdiff1d(b4bounce, aftbounce + 1)
        temp2 = np.setdiff1d(b4bounce, aftbounce - 1)
        abnormal_bounce_indices = np.unique(np.append(temp, temp2))
        normal_bounce = np.setdiff1d(b4bounce, abnormal_bounce_indices)

        #Bounce after hitting wall. Should be (2,0),(0,1),(1,0)
        #The returned indices should be inserted into the complete arrays
        # The index refers to the number of the before bounce.
        # To get after bounce + 1. B4 bounce -1
        temp3 = np.setdiff1d(b4bounce, aftwall + 1)
        #temp2 is the same as above
        abnormal_bounce_indices2 = np.unique(np.append(temp3, temp2))
        aftwall_bounce = np.setdiff1d(b4bounce, abnormal_bounce_indices2)


        t_normal_bounce = np.append(t_normal_bounce,t[normal_bounce] - t[normal_bounce - 1])
        t_aftwall_bounce = np.append(t_aftwall_bounce, t[aftwall_bounce]-t[aftwall_bounce - 1])

        x_normal = np.append(x_normal, x[normal_bounce])
        vx_normal = np.append(vx_normal, vx[normal_bounce])
        x_aftwall = np.append(x_aftwall, x[aftwall_bounce])
        vx_aftwall = np.append(vx_aftwall, vx[aftwall_bounce])
        dvx_normal = np.append(dvx_normal, vx[normal_bounce + 1] - vx[normal_bounce])
        dvx_aftwall = np.append(dvx_aftwall, vx[aftwall_bounce + 1] - vx[aftwall_bounce])
        vrw_normal = np.append(vrw_normal, vrw[normal_bounce])
        vrw_aftwall = np.append(vrw_aftwall, vrw[aftwall_bounce])


    # filter on t_bounce
    time_filter_aft = t_aftwall_bounce > 0.005
    time_filter_norm = t_normal_bounce > 0.005
    t_normal = t_normal_bounce[time_filter_norm]
    vx_normal = vx_normal[time_filter_norm]
    vx_aftwall = vx_aftwall[time_filter_aft]
    dvx_normal = dvx_normal[time_filter_norm]
    dvx_aftwall = dvx_aftwall[time_filter_aft]
    time_indices = np.argwhere(time_filter_norm)
    time_indices_catch = np.argwhere(~time_filter_norm)

    vrw_normal = vrw_normal[time_filter_norm]
    vrw_aftwall = vrw_aftwall[time_filter_aft]

    binsnorm, freqnorm = hist_plot(vrw_normal, range=(-3,3))
    binswall, freqwall = hist_plot(vrw_aftwall, range=(-3,3))
    plt.show()

    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_vwrhistnormal.txt', np.c_[binsnorm, freqnorm])
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_vwrhistwall.txt', np.c_[binswall, freqwall])