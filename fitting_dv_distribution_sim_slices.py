
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
    #plt.figure(fig_num)
    #plt.plot(bins, freq, marker)
    #plt.xlabel = xlabel
    #plt.ylabel = ylabel
    #plt.title = title
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
    vrw_normal = np.array([])
    vrw_aftwall = np.array([])
    t_normal_bounce = np.array([])
    t_aftwall_bounce = np.array([])
    w_normal = np.array([])
    dw_normal = np.array([])

    t_expt = np.array([])
    x_expt = np.array([])
    v_expt = np.array([])
    dv_expt = np.array([])
    vrw_expt = np.array([])


    for filename in file_iterator:#[filename, filename2, filename3]
        print(filename)
        data = np.loadtxt(filename)
        # Complete arrays
        t = np.array(data[:, 0])
        x = np.array(data[:, 1])
        vx = np.array(data[:, 2])
        vrw = np.array(data[:, 3])
        w = vx/vrw


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

        #For comparison with experiment
        all_bounces = np.append(normal_bounce, aftwall_bounce)





        t_normal_bounce = np.append(t_normal_bounce,t[normal_bounce] - t[normal_bounce - 1])
        t_aftwall_bounce = np.append(t_aftwall_bounce, t[aftwall_bounce]-t[aftwall_bounce - 1])

        x_normal = np.append(x_normal, x[normal_bounce])
        w_normal = np.append(w_normal, w[normal_bounce])
        vx_normal = np.append(vx_normal, vx[normal_bounce])
        x_aftwall = np.append(x_aftwall, x[aftwall_bounce])
        vx_aftwall = np.append(vx_aftwall, vx[aftwall_bounce])
        dw_normal = np.append(dw_normal,w[normal_bounce+1]-w[normal_bounce])
        dvx_normal = np.append(dvx_normal, vx[normal_bounce + 1] - vx[normal_bounce])
        dvx_aftwall = np.append(dvx_aftwall, vx[aftwall_bounce + 1] - vx[aftwall_bounce])
        vrw_normal = np.append(vrw_normal, vrw[normal_bounce])
        vrw_aftwall = np.append(vrw_aftwall, vrw[aftwall_bounce])

        t_expt = np.append(t_expt, t[all_bounces]-t[all_bounces-1])
        x_expt = np.append(x_expt, x[all_bounces])
        v_expt = np.append(v_expt, vx[all_bounces])
        dv_expt = np.append(dv_expt, vx[all_bounces+1]-vx[all_bounces])
        vrw_expt = np.append(vrw_expt, vrw[all_bounces])

    print('test')
    print(np.size(t_normal_bounce))
    print(np.size(t_aftwall_bounce))
    print(np.size(vx_normal))
    # filter on t_bounce
    time_filter_aft = t_aftwall_bounce > 0.005
    time_filter_norm = t_normal_bounce > 0.005
    t_normal = t_normal_bounce[time_filter_norm]
    w_normal = w_normal[time_filter_norm]
    vx_normal = vx_normal[time_filter_norm]
    vx_aftwall = vx_aftwall[time_filter_aft]
    dvx_normal = dvx_normal[time_filter_norm]
    dvx_aftwall = dvx_aftwall[time_filter_aft]
    dw_normal = dw_normal[time_filter_norm]
    time_indices = np.argwhere(time_filter_norm)
    time_indices_catch = np.argwhere(~time_filter_norm)
    dvrw_normal = vrw_normal[time_indices[:-2] + 1] - vrw_normal[time_indices[:-2]]

    vrw_normal_catch = vrw_normal[~time_filter_norm]
    vrw_normal = vrw_normal[time_filter_norm]


    vrw_aftwall = vrw_aftwall[time_filter_aft]
    x_normal = x_normal[time_filter_norm]
    x_aftwall = x_aftwall[time_filter_aft]

    all_dv = np.append(dvx_normal, dvx_aftwall)
    all_vx = np.append(vx_normal, vx_aftwall)
    all_x= np.append(x_normal, x_aftwall)

    #scatter_plot(-vrw_normal[1:],
    #             -vrw_normal[:-1], fig_num=13)
    #hist_plot(-vrw_normal, fig_num=10, range=(-2, 3))
    #hist_plot(-dvrw_normal, fig_num= 12, range=(-2,3))



    #hist_plot(vrw_normal_catch, fig_num=15, range=(-1,3),numbins=300)

    vrw_bins = np.array([])

    x_filter = np.abs(x_normal -0.015) < 5
    filter = (w_normal > vx_normal)
    nan_indices = np.isnan(dw_normal)
    #scatter_plot(vx_normal[x_filter& (~nan_indices)], dw_normal[(x_filter)& (~nan_indices)],'rx',fig_num=2)
    a = fit_data(dvx_normal[x_filter& (~nan_indices)], dw_normal[(x_filter)& (~nan_indices)])
    a = fit_data(vx_normal[x_filter & (~nan_indices)],
                  w_normal[(x_filter) & (~nan_indices)])
    #plt.show()
    #scatter_plot(vx_normal, w_normal,'bx',fig_num=3)

    #scatter_plot(dvx_normal[x_filter ], dw_normal[x_filter ], 'bx', fig_num=4)

    dvbinedges = np.arange(np.min(dvx_normal),np.max(dvx_normal),0.0005)
    mean_dw = []
    for i,h in enumerate(dvbinedges[:-1]):
        indices = np.argwhere((dvx_normal>dvbinedges[i])&(dvx_normal<dvbinedges[i+1]))
        mean_dw.append(np.nanmean(dw_normal[indices]))
    mean_dw=np.array(mean_dw)
    #scatter_plot(dvbinedges[:-1],mean_dw,'rx',fig_num=6)





    grads=[]
    binedges = np.arange(0,11,1)/1000
    print(binedges)
    for i in range(np.size(binedges)-1):
        filters = (np.abs(all_x-0.015) < binedges[i + 1]) & (
                    np.abs(all_x-0.015) > binedges[i])
        a = fit_data(all_vx[filters], all_dv[(filters)])

        grads.append(a)

    binedges=np.array(binedges)
    print('binedges')
    print(binedges)
    bins =  10 - 0.5*(binedges[:-1]+binedges[1:])*1000
    print(bins)
    grads=np.array(grads)
    print(grads)
    plt.figure(7)
    plt.plot(bins, grads, 'rx')
    plt.show()
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_grads.txt',np.c_[bins, grads])

    '''

    #Plotting simulation data
    hist_plot(t_normal_bounce, fig_num=1, numbins=300, range=(0,0.05), xlabel='time (s)', ylabel='p(t)', title='tbounce')
    #plt.savefig(output_path + output_file + '_tbounce.png')


    scatter_plot(vx_normal,dvx_normal, fig_num=2)
    if np.size(vx_aftwall)>50:
        scatter_plot(vx_aftwall, dvx_aftwall, marker='bx', fig_num=2, xlabel='vx', ylabel='dvx', title='friction')
    #plt.savefig(output_path + output_file + '_v_vs_dv.png')
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_vx_normal.txt',vx_normal)
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_dvx_normal.txt',dvx_normal)

    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_vx_aftwall.txt', vx_aftwall)
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_dvx_aftwall.txt', dvx_aftwall)

    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_dvx_normal.txt', vrw_normal)
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_vrw_aftwall.txt', vrw_aftwall)
    vrw_total=np.append(vrw_normal, vrw_aftwall)
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_dvx_total.txt', vrw_total)

    print('fractions:')



    f_normal = Fit('linear',x=vx_normal, y=dvx_normal,xlabel='vx', ylabel='dvx')
    f_normal.add_params(guess=[0, -0.1])
    f_normal.fit()
    if np.size(vx_aftwall > 50):
        f_wall = Fit('linear', x=vx_aftwall, y=dvx_aftwall, xlabel='vx',ylabel='dvx')
        f_wall.add_params(guess=[-0.1, 0],lower=[None, 'Fixed'], upper=[None, 'Fixed'])
        f_wall.fit()


    hist_plot(dvx_normal, fig_num=3, range=(-0.1, 0.1))
    if np.size(dvx_aftwall > 50):
       hist_plot(dvx_aftwall, marker='b-', fig_num=3, range=(-0.1, 0.1), xlabel='dvx', ylabel='p(dvx)', title='dvx')
    #plt.savefig(output_path + output_file + '_dvx.png')

    bins, freq = hist_plot(vrw_normal, fig_num=4, range=(-2, 3))
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_vrw_normal.txt',np.c_[bins,freq])

    if np.size(vrw_aftwall > 50):
        bins, freq = hist_plot(vrw_aftwall, marker='b-', fig_num=4, range=(-2, 3), xlabel='vrw', ylabel='p(vrw)', title='vrw')
        np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_3.25sim_vrw_aftwall.txt',np.c_[bins, freq])
    plt.savefig(output_path + output_file + '_vrw.png')


    
    #Simulating experimental plots
    print('simulate expt')

    nearwall = (np.abs(x_expt - 0.015) > 0.005)

    v_nearwall = v_expt[nearwall]

    t_nearwall = t_expt[nearwall]
    t_notnearwall = t_expt[~nearwall]

    #Filters out catch data
    t_filter = t_expt > 0.01

    v_notnearwall = v_expt[(~nearwall) & t_filter]
    dv_nearwall = dv_expt[nearwall & t_filter]
    dv_notnearwall = dv_expt[(~nearwall) & t_filter]
    vrw_nearwall = vrw_expt[nearwall & t_filter]
    vrw_notnearwall = vrw_expt[(~nearwall) & t_filter]

    t_normal_bounce = t_expt[t_filter]

    numpts = 535
    num_nearwall = np.sum(nearwall.astype(int))
    num_notnearwall = np.sum((~nearwall).astype(int))

    numb_near = int(numpts*num_nearwall/(num_nearwall+num_notnearwall))
    numb_notnear = int(numpts * num_notnearwall / (num_nearwall + num_notnearwall))

    hist_plot(t_normal_bounce[:numpts], fig_num=5, numbins=19, range=(0, 0.05), xlabel='time (s)', ylabel='p(t)', title='tbounce')
    plt.savefig(output_path + output_file + '_tbounce500.png')

    f_normal = Fit('linear', x=v_notnearwall[:numb_notnear], y=dv_notnearwall[:numb_notnear], xlabel='vx',
                   ylabel='dvx')
    f_normal.add_params(guess=[0, -0.1])
    f_normal.fit()
    if np.size(v_nearwall[:numb_near]) > 5:
        f_wall = Fit('linear', x=v_nearwall[:numb_near], y=dv_nearwall[:numb_near], xlabel='vx',
                     ylabel='dvx')
        f_wall.add_params(guess=[-0.1, 0],lower=[None, 'Fixed'], upper=[None, 'Fixed'])
        f_wall.fit()

    a_near, b_near = f_wall.fit_params
    a_notnear, b_notnear = f_normal.fit_params

    scatter_plot(v_notnearwall[:numb_notnear], dv_notnearwall[:numb_notnear],
                 fig_num=6)
    if np.size(v_nearwall[:numb_near]) > 5:
        scatter_plot(v_nearwall[:numb_near], dv_nearwall[:numb_near],
                     marker='bx', fig_num=6, xlabel='vx', ylabel='dvx',
                     title='tbounce')
        xfit = np.array([np.min(v_notnearwall), np.max(v_notnearwall)])
        yfitnear = xfit*a_near + b_near
        yfitnotnear = xfit*a_notnear + b_notnear
        plt.plot(xfit,yfitnear,'r-')
        plt.plot(xfit, yfitnotnear, 'b-')
    plt.savefig(output_path + output_file + '_v_vs_dvexptsim500.png')

    hist_plot(dv_notnearwall[:numb_notnear],numbins=11, fig_num=7, range=(-0.1, 0.1))
    if np.size(v_nearwall[:numb_near]) > 5:
        hist_plot(dv_nearwall[:numb_near], numbins=11, marker='b-', fig_num=7, range=(-0.1, 0.1), xlabel='dvx', ylabel='p(dvx)', title='dvx')
    plt.savefig(output_path + output_file + '_dvxexptsim500.png')

    hist_plot(vrw_notnearwall[:numb_notnear], numbins=11, fig_num=8, range=(-2, 3))
    #if np.size(v_nearwall[:numb_near]) > 5:
    hist_plot(vrw_nearwall[:numb_near], marker='b-', numbins=11,fig_num=8, range=(-2, 3), xlabel='vrw', ylabel='p(vrw)', title='vrw')
    plt.savefig(output_path + output_file + '_vrwexptsim500.png')


    #plt.show()
    '''