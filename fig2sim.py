
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

    x_normal = x_normal[time_filter_norm]
    x_aftwall = x_aftwall[time_filter_aft]

    all_dv = np.append(dvx_normal, dvx_aftwall)
    all_vx = np.append(vx_normal, vx_aftwall)
    all_x= np.append(x_normal, x_aftwall)

    x_filter = np.abs(x_normal -0.015) < 5

    scatter_plot(vx_normal, dvx_normal)
    scatter_plot(vx_aftwall, dvx_aftwall)
    plt.show()

    bins, freq = hist_plot(vx_normal)

    #Data for figure 2c need dv and vx for all points in 2 files. one for those that collided with wall and one for the rest
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/vdvnormal.txt',np.c_[vx_normal, dvx_normal])
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/vdvaftwall.txt',np.c_[vx_aftwall, dvx_aftwall])


    #Fig2d plot variation of gradient with distance from wall
    grads=[]
    binedges = np.arange(0,11,1)/1000
    print(binedges)
    for i in range(np.size(binedges)-1):
        filters = (np.abs(all_x-0.015) < binedges[i + 1]) & (
                    np.abs(all_x-0.015) > binedges[i])
        a = fit_data(all_vx[filters], all_dv[(filters)])

        grads.append(a)

    binedges=np.array(binedges)
    bins =  10 - 0.5*(binedges[:-1]+binedges[1:])*1000
    grads=np.array(grads)
    plt.figure(7)
    plt.plot(bins, grads, 'rx')
    plt.show()
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_2.25sim_grads.txt',np.c_[bins, grads])

