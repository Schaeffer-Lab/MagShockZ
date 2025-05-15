# This file definitely should either be deleted or merged into another file

import yt

def pull_value_at_coord(coordinate: list, field: str, ds: yt.data_objects.static_output.Dataset,plot=True):
    """Function to extract the value of a field at a specific coordinate in the dataset,
       also plots said coordinate on a slice plot in z
       Args:
        coordinate (list): list of coordinates [x,y,z] in code units
        field (str): field to extract value from
        ds (yt.data_objects.static_output.Dataset): dataset to extract value from"""
    
    slc = yt.SlicePlot(ds,"z",(field),center=[0.0, 0.4, 0.0]).zoom(1.4)
    
    # Convert coordinates to dataset units
    coord = ds.arr(coordinate, 'code_length')

    # Extract the value at the specific coordinates
    value = ds.find_field_values_at_point(fields=('flash',field), coords=coord)
    print(f"{field} at {coord}: {value}")

    slc.annotate_marker(coord)
    if plot: slc.show()
    return value