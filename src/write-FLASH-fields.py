############################
#### UNFINISHED AS FUCK ####
############################


import numpy as np
from pathlib import Path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("FLASH_DATA", type=str, help="Path to the FLASH data file")
    parser.add_argument("INPUT_FILE_NAME", type=str, help="Name of the input file")
    parser.add_argument("--normalizations", type=str, help="Path to the normalizations file")
    args = parser.parse_args()
    with open(args.normalizations, "r") as f:
        normalizations = {k: float(v) for k, v in [line.split() for line in f]}
    main(args.FLASH_DATA, args.INPUT_FILE_NAME, normalizations)

def save_slice(normalizations:dict, target_index:int=84):
    '''
    args: normalizations: dict, target_index: int
    normalizations: dict
        key: str, field name
        value: float, normalization factor. Note: function automatically divides by this factor
    target_index: int
        index of the edge of the target

    Note: Density data will be output as a numpy array because OSIRIS uses its own interpolator for density data
    '''
    import multiprocessing as mp
    import pickle
    from scipy.interpolate import RegularGridInterpolator
    
    interp_dir = output_dir / "interp"
    if not (interp_dir).exists():
        (interp_dir).mkdir()
        
    z_middle_index=dims[2]//2

    chunk_size = 100  # Adjust this based on your memory constraints


    for (field, normalization) in normalizations.items():
        field_data = np.zeros(all_data['flash', field][:, target_index:, z_middle_index].shape)
        for i in range(0, all_data['flash', field].shape[0], chunk_size):
            end = min(i + chunk_size, all_data['flash', field].shape[0])
            field_data_chunk = np.array(all_data['flash', field][i:end, target_index:, z_middle_index]) / normalization
            field_data[i:end, :] = field_data_chunk

        if field in ['edens', 'aldens', 'mgdens']:
            lower_bound = 0.001
            field_data[field_data < lower_bound] = 0
            np.save(f"{interp_dir}/{field}.npy", field_data)
        else:
            x = all_data['flash', 'x'][:, 0, 0] * omega_pe / c
            y = all_data['flash', 'y'][0, target_index:, 0] * omega_pe / c
            interp1 = RegularGridInterpolator((y, x), field_data.T, method='linear', bounds_error=False, fill_value=0)
            with open(f"{interp_dir}/{field}.pkl", "wb") as f:
                pickle.dump(interp1, f)

    # with mp.Pool(len(normalizations)) as pool:
    #     pool.map(process_field, normalizations.items())


def main(FLASH_DATA: str, INPUT_FILE_NAME: str, normalizations: dict) -> None:
    from load_derived_FLASH_fields import derive_fields
    ds = derive_fields(FLASH_DATA,rqm=100)
    # this is from the yt documentation

    level = 2
    dims = ds.domain_dimensions * ds.refine_by**level

    # We construct an object that describes the data region and structure we want
    # In this case, we want all data up to the maximum "level" of refinement
    # across the entire simulation volume.  Higher levels than this will not
    # contribute to our covering grid.

    all_data = ds.covering_grid(
        level,
        left_edge=[-0.6, -0.075, -0.6],
        dims=dims,
    )

    output_dir = Path(f"/home/dschneidinger/MagShockZ/input_files/{INPUT_FILE_NAME}") # Fix this
    if not output_dir.exists():
        output_dir.mkdir()



FLASH_DATA = "~/shared/data/OSIRIS_transfer/MAGON/MagShockZ_hdf5_chk_0005"
# FLASH_DATA = "~/cellar/VAC_DEREK3D_20um/MagShockZ_hdf5_chk_0003"
INPUT_FILE_NAME = "magshockz-v2.0.2d"

# This can take a while and always seems to crash the kernel if you try to do more than 1 or 2. Uncomment as needed