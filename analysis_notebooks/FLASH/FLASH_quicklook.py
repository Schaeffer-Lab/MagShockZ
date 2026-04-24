import yt
from pathlib import Path
import os

def timeseries(field):
    os.makedirs(f"{field}_plots", exist_ok=True)
    for i in range(0, 20):
        t = i
        FLASH_data = Path("/pscratch/sd/d/dschnei/FLASH_3D_noshield/MagShockZ_hdf5_plt_cnt_" + str(t).zfill(4))

        ds = yt.load(FLASH_data)

        slc = yt.SlicePlot(ds, "z", field)
        slc.set_cmap(field, "Reds")
        slc.annotate_timestamp(corner="upper_left", redshift=True, draw_inset_box=True)
        slc.save(f"/pscratch/sd/d/dschnei/MagShockZ/analysis_notebooks/FLASH/{field}_plots/{field}_{str(t).zfill(4)}.png")

if __name__ == "__main__":
    timeseries("density")
    timeseries("magx")
    timeseries("magy")
    timeseries("magz")
    FLASH_data = Path("/pscratch/sd/d/dschnei/FLASH_3D_noshield/MagShockZ_hdf5_chk_0004")

    ds = yt.load(FLASH_data)

    slc = yt.SlicePlot(ds, "x", "magz")
    # slc.set_cmap("magz", "Reds")
    slc.swap_axes()
    slc.save(f"/pscratch/sd/d/dschnei/MagShockZ/analysis_notebooks/FLASH/magz_plots/magz_new.png")

    slc = yt.SlicePlot(ds, "x", "density")
    # slc.set_cmap("density", "Reds")
    slc.swap_axes()
    slc.save(f"/pscratch/sd/d/dschnei/MagShockZ/analysis_notebooks/FLASH/density_plots/density_x_slice.png")

    slc = yt.SlicePlot(ds, "z", "density")

    # slc.set_cmap("density", "Reds")
    slc.save(f"/pscratch/sd/d/dschnei/MagShockZ/analysis_notebooks/FLASH/density_plots/density_z_slice.png")