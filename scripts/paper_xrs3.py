import matplotlib.pyplot as plt
import numpy as np

ene_axis = np.loadtxt("/mnt/cellar/shared/data/Z_MagShockZ-A1340A_07-2024/XRS3/z4013energyaxis_xrs3.csv",
                delimiter=',')
spatial_axis = np.loadtxt("/mnt/cellar/shared/data/Z_MagShockZ-A1340A_07-2024/XRS3/z4013spatialaxis_xrs3.csv",
                delimiter=',')
dat = np.loadtxt("/mnt/cellar/shared/data/Z_MagShockZ-A1340A_07-2024/XRS3/z4013_xrs3.csv",
                delimiter=',')

ene_idx_bounds = [0, -1]
space_idx_bounds = [200, -1]
# Slice data and axes
dat_slice = np.nan_to_num(dat, nan=0.001)  # Replace NaNs with 0
dat_slice[dat_slice < 0] = 0.001  # Ensure no negative values
dat_slice = np.log(dat_slice[space_idx_bounds[0]:space_idx_bounds[1], ene_idx_bounds[0]:ene_idx_bounds[1]])


plt.imshow(dat_slice, aspect='auto', extent=[ene_axis[ene_idx_bounds[0]], ene_axis[ene_idx_bounds[1]], spatial_axis[space_idx_bounds[0]], spatial_axis[space_idx_bounds[1]]], cmap='hot',vmin = 2.5)
plt.xlabel('Energy Axis (keV)')
plt.ylabel('Spatial Axis (?)')
plt.colorbar(label='Intensity (a.u.)')

plt.savefig("/home/dschneidinger/MagShockZ/plots_for_paper/XRS3_plot.png",)