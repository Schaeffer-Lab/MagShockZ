import osh5def
import osh5vis
import osh5io
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Set the default font size for axis labels
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
# Set the default font size for tick labels
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

t = 140
ylims = (0, 5)


data_dir = Path('../simulations/save_data/magshockz-v3.2.1d-7.14debye/MS')
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
# interface = pickle.load(open(f'{data_dir.parent}/instance.pkl', 'rb'))

channel_density = (data_dir / 'DENSITY' / 'channel' / 'charge').glob('*.h5')
channel_density = sorted(channel_density)

data = osh5io.read_h5(channel_density[t].as_posix())
print(data.shape)

osh5vis.osplot(data,c='blue',xlim=(3000,4300), ylim = ylims, label = '90 degrees')





data_dir = Path('../simulations/save_data/magshockz-v3.2.1d-60degrees/MS')
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

channel_density = (data_dir / 'DENSITY' / 'background' / 'charge').glob('*.h5')
channel_density = sorted(channel_density)

data = osh5io.read_h5(channel_density[t].as_posix())
print(data.shape)

osh5vis.osplot(data,c='green',xlim=(3000,4300),ylim = ylims, label = '60 degrees')




data_dir = Path('../simulations/save_data/magshockz-v3.2.1d-45degrees.1d/MS')
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

channel_density = (data_dir / 'DENSITY' / 'background' / 'charge').glob('*.h5')
channel_density = sorted(channel_density)

data = osh5io.read_h5(channel_density[t].as_posix())
print(data.shape)

osh5vis.osplot(data,c='red',xlim=(2750,4000),ylim = ylims, label = '45 degrees')



