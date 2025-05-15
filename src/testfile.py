import yt

yt.enable_plugins()

ds = yt.load_for_osiris(filename = "~/shared/data/VAC_DEREK3D_20um/MagShockZ_hdf5_chk_0006",rqm = 100, B_background = 75000)

level = 0
dims = ds.domain_dimensions * ds.refine_by**level

# We construct an object that describes the data region and structure we want
# In this case, we want all data up to the maximum "level" of refinement
# across the entire simulation volume.  Higher levels than this will not
# contribute to our covering grid.

all_data = ds.covering_grid(
    level,
    left_edge=ds.domain_left_edge,
    dims=dims,
    num_ghost_zones=1,
)
all_data
import matplotlib.pyplot as plt
plt.imshow(all_data['flash',"idens"][:,:,(all_data['flash',"idens"].shape[0])//2])

yt.SlicePlot(ds, "z", ("flash", "sidens"), center = ds.domain_center).save("sidens.png")
yt.SlicePlot(ds, "z", ("flash", "channeldens"), center = ds.domain_center).save("channel.png")
yt.SlicePlot(ds, "z", ("flash", "sheathedens"), center = ds.domain_center).save("sheathe.png")
yt.SlicePlot(ds, "z", ("flash", "backgrounddens"), center = ds.domain_center).save("background.png")
yt.SlicePlot(ds, "z", ("flash", "soliddens"), center = ds.domain_center).save("solid.png")
