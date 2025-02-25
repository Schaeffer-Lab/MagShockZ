import numpy as np
import yt
import yt.units as u

def add_current_density(ds):

    c = u.speed_of_light_cgs
    conv_factor = c/(4*np.pi)
    
    ds.add_gradient_fields(('flash', 'magx'))
    ds.add_gradient_fields(('flash', 'magy'))
    ds.add_gradient_fields(('flash', 'magz'))

    def make_Jx(field, data):
        return conv_factor * (data["flash","magz_gradient_y"] - data["flash","magy_gradient_z"])
    def make_Jy(field, data):
        return conv_factor * (data["flash","magx_gradient_z"] - data["flash","magz_gradient_x"])
    def make_Jz(field, data):
        return conv_factor * (data["flash","magy_gradient_x"] - data["flash","magx_gradient_y"])
    

    ds.add_field(('flash', 'Jx'), function=make_Jx, units='code_magnetic/code_time', sampling_type='cell')
    ds.add_field(('flash', 'Jy'), function=make_Jy, units='code_magnetic/code_time', sampling_type='cell')
    ds.add_field(('flash', 'Jz'), function=make_Jz, units='code_magnetic/code_time', sampling_type='cell')

    ds.force_periodicity()
    return ds


def add_vi_and_ve(ds):


    """
    Add ion and electron velocities to the dataset.

    Parameters:
    ds : yt.Dataset
    """
    
    e = u.electron_charge_cgs
    m_e = u.mass_electron_cgs

    def _v_ix(field, data):
        return ((m_e*data['flash','Jx']/(data['flash','dens']*e)) + data['flash','velx'])
    def _v_iy(field, data):
        return ((m_e*data['flash','Jy']/(data['flash','dens']*e)) + data['flash','vely'])
    def _v_iz(field, data):
        return ((m_e*data['flash','Jz']/(data['flash','dens']*e)) + data['flash','velz'])
    
    def _v_ex(field, data):
        return ((m_e*data['flash','Jx']/(data['flash','dens']*e) - data['flash','Jx']/(data['flash','edens']*e)) + data['flash','velx'])
    def _v_ey(field, data):
        return ((m_e*data['flash','Jy']/(data['flash','dens']*e) - data['flash','Jy']/(data['flash','edens']*e)) + data['flash','vely'])
    def _v_ez(field, data):
        return ((m_e*data['flash','Jz']/(data['flash','dens']*e) - data['flash','Jz']/(data['flash','edens']*e)) + data['flash','velz'])
    
    # Update the field definitions
    ds.add_field(('flash', 'v_ix'), function=_v_ix, units='code_velocity', sampling_type='cell')
    ds.add_field(('flash', 'v_iy'), function=_v_iy, units='code_velocity', sampling_type='cell')
    ds.add_field(('flash', 'v_iz'), function=_v_iz, units='code_velocity', sampling_type='cell')
    ds.add_field(('flash', 'v_ex'), function=_v_ex, units='code_velocity', sampling_type='cell')
    ds.add_field(('flash', 'v_ey'), function=_v_ey, units='code_velocity', sampling_type='cell')
    ds.add_field(('flash', 'v_ez'), function=_v_ez, units='code_velocity', sampling_type='cell')

    return ds


# from pathlib import Path
# plot_path = "~/shared/data/VAC_DEREK3D_20um/MagShockZ_hdf5_chk_0028"
# from load_derived_FLASH_fields import derive_fields

# ds = derive_fields(plot_path,rqm=100,ion_2='Si')
# ds = add_current_density(ds)


# ds = add_vi_and_ve(ds)

# yt.SlicePlot(ds, 'z', ('flash', 'v_iy')).save('v_iy.png')