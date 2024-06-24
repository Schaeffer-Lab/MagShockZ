# Run Imports
import numpy as np
from vysxd_define import *
from vysxd_analysis import *


def FPC(directory):
    p1x1_files = np.sort(os.listdir(f'{directory}/MS/PHA/p1x1/electrons/'))
    p1x1 = vysxd_get_data(f'{directory}/MS/PHA/p1x1/electrons/p1x1-electrons-000000.h5')
    A_result = np.empty((len(p1x1.Y),len(p1x1.X)-2,len(p1x1_files)-1))
    B_result = np.empty((len(p1x1.Y),len(p1x1.X)-2,len(p1x1_files)-1))
    for i in range(len(p1x1_files)-1):
        p1x1 = vysxd_get_data(f'{directory}/MS/PHA/p1x1/electrons/' + p1x1_files[i])

        e1_D_xt = get_osiris_quantity_1d(f'{directory}/MS/FLD/e1/')[0]

        e1_D_xt = e1_D_xt[i][1:-1]
        e1_D_xt = np.array([e1_D_xt]*len(p1x1.Y))

        dfdv = np.gradient(p1x1.DATA[:,1:-1],axis=0)
        dfdr = np.gradient(p1x1.DATA[:,1:-1],axis=1)

        e_times_dfdv = np.multiply(e1_D_xt,dfdv)

        vsquared_arr = np.transpose(np.array([p1x1.Y**2]*len(p1x1.X[1:-1])))
        vcubed_arr = np.transpose(np.array([p1x1.Y**3]*len(p1x1.X[1:-1])))

        A = 1/2*np.multiply(vcubed_arr,dfdr)
        B = -1/2*np.multiply(vsquared_arr, e_times_dfdv)

        A_result[:,:,i] = A
        B_result[:,:,i] = B
    return A_result, B_result