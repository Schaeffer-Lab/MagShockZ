## IMPORTS ##
import numpy as np
from pathlib import Path
import yt
import numpy as np
import matplotlib.pyplot as plt

## Path to FLASH data. This data was chosen because it's generally a simple setup.
## We want to characterize the fundamental behavior of a pison expanding out into a magnetized background. 
data_path = Path("/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_06-2024/MAGON/MagShockZ_hdf5_chk_0005")

# Use the plugin I built for yt to load in FLASH data and split up ion species.
yt.enable_plugins()

ds = yt.load_for_osiris(data_path, rqm_factor = 10, ion_mass_thresholds = [28,35], rqm_thresholds= [4500,7100])

slc = yt.SlicePlot(ds, 'z', 'channeldens')
slc.save("simplified_magshockz_figs/fig.png")

print("'ello")