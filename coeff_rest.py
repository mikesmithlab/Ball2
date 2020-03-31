import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import csv
#from itertools import izip
from Generic.filedialogs import get_files_directory
#from OSFile import get_files_directory, write_row_to_csv
from Generic.fitting import Fit


def diffs(data, min_val=0.01):
    yvel = data['yVelMM'].values
    yvel_threshold = np.abs(data['yVelMM']).mean()/2
    vx = data['xVelMM'].values

    # yvel - surface vel
    rel_vel = (data['yVelMM']- 500*data['surfaceHeightMM'].diff()).values
    ang_vel = data['omega_k'].values
    bounce_indices = data[data['bounce'] == True].index
    #plt.figure()
    #plt.plot(data['ballHeightMM'].index, data['ballHeightMM'], 'ro-')
    #plt.plot(data['surfaceHeightMM'].index, data['surfaceHeightMM'],'bo-')
    #plt.show()
    max_index = np.array([]).astype(int)
    min_index = np.array([]).astype(int)
    for index,dummy in enumerate(bounce_indices[:-1]):
        max_index = np.append(max_index, np.argmax(yvel[int(bounce_indices[index]):int(bounce_indices[index + 1])]) + int(bounce_indices[index]))
        min_index = np.append(min_index, np.argmin(yvel[int(bounce_indices[index]):int(bounce_indices[index + 1])]) + int(bounce_indices[index]))

    pos_acc_indices = data[(data['surfaceHeightMM'] < 0) & (np.abs(data['yVelMM']) > yvel_threshold)].index

    min_index = min_index[:-1]
    max_index = max_index[1:]


    #print(bounce_indices)
    #plt.plot(bounce_indices, 0.5*np.ones(np.shape(bounce_indices)),'gx')
    #plt.plot(max_index, yvel[max_index], 'bo')
    #plt.plot(min_index, yvel[min_index], 'bo')
    #plt.show()





    b4_relv = np.array([]).astype(int)
    after_relv = np.array([]).astype(int)
    b4_angv = np.array([]).astype(int)
    after_angv = np.array([]).astype(int)
    b4_vx = np.array([])
    after_vx = np.array([])
    for index, indexval in enumerate(min_index):
        if indexval in pos_acc_indices:
            b4_angv = np.append(b4_angv,ang_vel[indexval])
            b4_relv = np.append(b4_relv,rel_vel[indexval])
            b4_vx = np.append(b4_vx,vx[indexval])
            after_angv = np.append(after_angv, ang_vel[max_index[index]])
            after_relv = np.append(after_relv,rel_vel[max_index[index]])
            after_vx = np.append(after_vx,vx[max_index[index]])

    #return b4_vx, after_vx - b4_vx
    #return b4_vx, after_vx
    return b4_relv, after_relv
    #return b4_angv, after_angv
    
if __name__ == '__main__':

    #Load dataframe
    #filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/',title='Select Data File', filetypes = (('DataFrames', '*finaldata.hdf5'),))    
    basepath='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/*'
    #basepath='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed8mm/*'
    #basepath = '/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed12_5mm/*'
    #basepath = '/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed_new10mm/*'
    #names = ['80_050']#,'80_062','120_077','120_062']#,'120_050','240_077','240_062','240_050','400_077','400_062','400_050']
    names = ['80_077']
    pathnames = [basepath + name + '_data_finaldata.hdf5' for name in names]
    param = 'xVelMM'
    #each path get the 3 files for a particular experiment.
    for path in pathnames:
        filenames=get_files_directory(path, full_filenames=True)
        b4 = np.array([])
        after = np.array([])

        filename_op = min(filenames)[:-15]
        print(filename_op)
        for i, filename in enumerate(filenames):
            
            
            data = pd.read_hdf(filename)
            b4_relv, after_relv = diffs(data) #, filename_op + '_forces_v', param)


            b4 = np.append(b4, b4_relv)

            after = np.append(after, after_relv)


        fit_obj = Fit('linear', xlabel='b4', ylabel='after')


        fit_obj.add_fit_data(x=b4[after>0],y=after[after>0])
        fit_obj.add_params(guess=[-0.1, 0], lower=[None,'Fixed'], upper=[None, 'Fixed'])
        #logic = np.abs(after-b4) < 20
        #fit_obj.add_filter(logic)
        #fit_obj.add_filter(logic)

        params = fit_obj.fit(interpolation_factor=0.1, errors=True)

        fit_obj.plot_fit(show=True)

        #plt.figure()
        #plt.plot(after,)
        #print(filename_op + '_coeff_rest' + '.csv')
        np.savetxt(filename_op + '_ang_coeff_rest' + '.csv', np.c_[-b4,after], fmt='%.5f', delimiter=',', header="-b4_relv,after_relv")


        
    
    
    
    
    
    
    