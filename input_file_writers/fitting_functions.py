import numpy as np
import matplotlib.pyplot as plt
import yt
import pwlf
import plasmapy
import astropy.units as u
import astropy.constants as const



class Ray:
    """
    Class to handle the lineouts and fitting from FLASH to OSIRIS.
    """
    def __init__(self, ds, start_pt, end_pt, reference_density = 5e18 * yt.units.cm**-3, rqm_factor = 50):
        if isinstance(ds, str):
            yt.enable_plugins()
            self.ds = yt.load_for_osiris(ds, rqm_factor=rqm_factor)
        elif isinstance(ds, yt.frontends.flash.data_structures.FLASHDataset):
            self.ds = ds
        self.rqm_factor = rqm_factor 

        # omega_pe = plasmapy.formulary.plasma_frequency(reference_density * u.cm**-3, particle="e-")
        # print(f"plasma frequency is {omega_pe:.2e} rad/s for reference density {reference_density:.2e} cm^-3")
        omega_pe = np.sqrt(reference_density * yt.units.electron_charge_mks**2 / (yt.units.eps_0 * yt.units.electron_mass)).to('1/s')
        print(f"plasma frequency is {omega_pe:.2e} rad/s for reference density {reference_density:.2e} cm^-3")
        B_norm = (omega_pe * yt.units.electron_mass * yt.units.speed_of_light / yt.units.elementary_charge).to('Gauss')
        v_norm = (yt.units.speed_of_light / np.sqrt(rqm_factor)).to('cm/s')
        E_norm = (omega_pe * yt.units.electron_mass * yt.units.speed_of_light / yt.units.elementary_charge / np.sqrt(rqm_factor)).to('statV/cm')

        # Gonna throw my work in here because I feel like I'm always redoing this.
        # We want T_e/T_i to be conserved, so
        # m_e * vthe^2/ (m_i vthi^2) = m_e' * vthe'^2 / (m_i' vthi'^2)
        # m_e * vthe^2/ (m_i vthi^2) = m_e' * vthe^2 / c^2 / (m_i' vthi'^2)
        # m_e / (m_i vthi^2) = m_e' / c^2 / (m_i' vthi'^2)
        # vthi'^2 = vthi^2 * (m_i/m_e) / (m_i'/m_e') / c^2
        # where rqm_factor is (m_i/m_e)/(m_i'/m_e')
        # vthi' = vthi * sqrt(rqm_factor) / c

        vth_ele_norm = yt.units.speed_of_light
        vth_ion_norm = (yt.units.speed_of_light / np.sqrt(self.rqm_factor)).to('cm/s')

        self.start_pt = start_pt
        self.end_pt = end_pt
        self.length = self._get_distance_axis()
        self.osiris_length = self.length / (yt.units.speed_of_light / omega_pe)  # Normalize to OSIRIS units
        self.osiris_length = (self.osiris_length - self.osiris_length[0]).value  # Center the x values to start at 0
        self.math_funcs = {}

        # Check that normalizations are done correctly
        B_test = 100_000 * yt.units.Gauss
        B_test_normalized = (B_test / B_norm).to(yt.units.dimensionless)
        print(f"Test magnetic field normalization: {B_test} normalizes to {B_test_normalized:.2f} in OSIRIS units")
        print(f"Corresponding to {5.681e-8 * B_test_normalized * omega_pe} Gauss in real units")
        print(f"corresponding to {3.204e-3 * B_test_normalized * np.sqrt(reference_density)} Gauss in real units")


        self.normalizations = {
            # Density normalizations
            'edens': reference_density, 'sidens': reference_density, 'aldens': reference_density,
            
            # Magnetic field normalizations
            'magx': B_norm, 'magy': B_norm, 'magz': B_norm,
            
            # Electric field normalizations
            'Ex': E_norm, 'Ey': E_norm, 'Ez': E_norm,
            
            # Velocity normalizations
            'v_ix': v_norm, 'v_iy': v_norm, 'v_iz': v_norm,
            'v_ex': v_norm, 'v_ey': v_norm, 'v_ez': v_norm,
            
            # Thermal velocity for electrons
            'vthele': vth_ele_norm,
            # Thermal velocity for ions
            'vthion': vth_ion_norm,
        }


    def _get_distance_axis(self):
        '''
        Helper function to get a distance axis for the ray on which you are taking a lineout in real units
        '''
        euclidean_length_of_ray = np.sqrt((self.end_pt[0]-self.start_pt[0])**2 + (self.end_pt[1]-self.start_pt[1])**2 + (self.end_pt[2]-self.start_pt[2])**2)
        dist_from_origin = np.sqrt(self.start_pt[0]**2 + self.start_pt[1]**2 + self.start_pt[2]**2)
        return np.array(self.ds.ray(self.start_pt, self.end_pt)['t']* euclidean_length_of_ray+dist_from_origin)

    def _get_field_values(self, field: str):
        '''
        Helper function to get the field values for the ray on which you are taking a lineout
        This handles the normalization as well

        Takes the square root of temperature fields to convert them to thermal velocities.
        '''
        ray = self.ds.ray(self.start_pt, self.end_pt)
        # For some god forsaken reason, yt does not return the values in the order of the ray, so we have to sort them
        ray_sort = np.argsort(ray["t"])

        values = (ray[(field)][ray_sort] / self.normalizations[field]).value
        return values

    def show_ray(self, field: str):
        """ Show the ray in a slice plot with some specified field."""
        slc = yt.SlicePlot(self.ds, "z", ("flash", field))
        slc.annotate_line(self.start_pt, self.end_pt, color="red")
        slc.show()

    def show_lineout(self, field: str, log: bool=False):
        """ Show the lineout of some field along ray"""
        line = yt.LinePlot(ds=self.ds, fields=("flash", field), start_point=self.start_pt, end_point=self.end_pt, npoints=10000)
        line.set_log(("flash", field), log=log)
        line.show()


    def fit(self, field, degree=3, left_value=None, right_value=None, fit_func = "polynomial", plot=False, dim_var="x2"):
        """
        Fit a polynomial to the lineout data.
        
        Parameters:
        -----------
        field : str
            Field name to fit
        degree : int
            Degree of polynomial or number of segments for piecewise
        left_value : float, optional
            Value to use for x < lineout start
        right_value : float, optional
            Value to use for x > lineout end
        fit_func : str
            Type of fit: 'polynomial', 'exponential', or 'piecewise'
        plot : bool
            Whether to plot the fit
        dim_var : str
            Dimension variable to use in OSIRIS math function ('x1', 'x2', or 'x3')
        """
        if field.endswith("dens"):
            print("suggest using fit_density for initializing density fields")

        # TODO make this way better, way too hard-coded rn
        vals = self._get_field_values(field)
        if dim_var == "x1" and field in ["v_ix", "v_ex", "Ex", "Bx"]:
            vals = -vals  # Account for coordinate rotation for x1 axis
        if fit_func == "polynomial":
            coefficients = np.polyfit(self.osiris_length, vals, degree)
            function = np.poly1d(coefficients)
            y_fit = function(self.osiris_length)

        # Right now, exponential fitting is not working correctly
        elif fit_func == "exponential":
            coefficients = np.polyfit(np.log(self.osiris_length), vals, 1) # TODO fix AI slop
            function = lambda x: np.exp(coefficients[0]) * np.exp(x + coefficients[1])
            y_fit = function(self.osiris_length)

        elif fit_func == "piecewise":
            # Note that for piecewise, the meaning of degree is different. It is the number of segments, not the degree of the polynomial.
            pwlf_model = pwlf.PiecewiseLinFit(self.osiris_length, vals)
            res = pwlf_model.fit(degree)
            y_fit = pwlf_model.predict(self.osiris_length)
            
        else:
            raise ValueError("Unsupported fit function. Use 'polynomial', 'exponential', or 'piecewise'.")

        if plot == True:
            # Plot the original data points
            plt.scatter(self.osiris_length, vals, label='Data Points')

            # Plot the polynomial fit
            plt.plot(self.osiris_length, y_fit, color='red', label=f'{degree} Degree {fit_func} Fit')

            plt.xlabel('dist [osiris_units]')
            plt.ylabel(f'{field} [osiris_units]')



        # OSIRIS FORMATTING
        precision = 4
        result = ''

        if fit_func == "polynomial":
            # In this case, the function can be defined in all space
            if left_value is None and right_value is None:
                for i in range(len(coefficients)):
                    result = f"{result}({np.format_float_scientific(coefficients[i], sign=False)})*({dim_var})^("+str(degree-i) + ") +"
                result = f"{result.strip(' +')}"

            # In this case, the function is only defined in the region of interest
            if left_value is not None and right_value is not None:
                result = f"{result}if({dim_var} < {round(self.osiris_length[0],precision)}, {left_value}, if({dim_var} < {round(self.osiris_length[-1], precision)}, "
                for i in range(len(coefficients)):
                    result = f"{result} ({np.format_float_scientific(coefficients[i], precision=precision, sign=False)})*({dim_var}^("+str(degree-i) + ") +"
                result = result.strip(' +')+f", {right_value}))"
            # In this case, the function starts at the left side of the box and goes to the right boundary
            if left_value is None and right_value is not None:
                result = f"{result}if({dim_var} < {round(self.osiris_length[-1],precision)}, "
                for i in range(len(coefficients)):
                    result = f"{result} ({np.format_float_scientific(coefficients[i], precision=precision, sign=False)})*({dim_var}^("+str(degree-i) + ") +"
                result = result.strip(' +')+f", {right_value})"
        if fit_func == "exponential":
            if left_value is None and right_value is None:
                result = f"{result}({np.format_float_scientific(coefficients[0], precision=precision, sign=False)})*exp({dim_var} + {np.format_float_scientific(coefficients[1], precision=precision, sign=False)})"

            # In this case, the function is only defined in the region of interest
            if left_value is not None and right_value is not None:
                result = f"{result}if({dim_var} < {round(self.osiris_length[0],precision)}, {left_value}, if({dim_var} < {round(self.osiris_length[-1], precision)}, "
                result = f"{result} ({np.format_float_scientific(coefficients[0], precision=precision, sign=False)})*exp({dim_var} + {np.format_float_scientific(coefficients[1], precision=precision, sign=False)})"

            # In this case, the function starts at the left side of the box and goes to the right boundary
            if left_value is None and right_value is not None:
                result = f"{result}if({dim_var} < {round(self.osiris_length[-1],precision)}, "
                result = f"{result} ({np.format_float_scientific(coefficients[0], precision=precision, sign=False)})*exp({dim_var} + {np.format_float_scientific(coefficients[1], precision=precision, sign=False)})"

        if fit_func == "piecewise":
            extra_parentheses = 0
            if left_value is not None:
                result += f"if({dim_var} < {round(pwlf_model.fit_breaks[0], precision)}, {left_value}, "
                extra_parentheses += 1

            for i in range(len(pwlf_model.fit_breaks)-1):
                result += f"if({dim_var} < {np.format_float_scientific(pwlf_model.fit_breaks[i+1],precision)}, "
                result += f"{dim_var}*({np.format_float_scientific(pwlf_model.slopes[i], precision)}) + ({np.format_float_scientific(pwlf_model.intercepts[i], precision)}), "

            result += str(pwlf_model.intercepts[-1] + pwlf_model.slopes[-1] * (self.osiris_length[-1] - pwlf_model.fit_breaks[-1])) + ")"*(len(pwlf_model.fit_breaks) - 1 + extra_parentheses) 

        # Save that ish for later so we can write an easy breezy beautiful input file
        self.math_funcs[field] = result
        return result

    def fit_density(self, field, n_skip=2, precision = 2, plot = False):
        """
        OSIRIS has a built-in piecewise linear density initialization, so let's take advantage of that
        If you are encountering errors with OSIRIS reading in the profile information, you likely have too many points - increase n_skip
        """
        vals = self._get_field_values(field)

        while len(vals) / n_skip > 64:
            n_skip += 1

        vals = vals[::n_skip]



        x_axis = np.linspace(self.osiris_length[0], self.osiris_length[-1], len(vals))

        # Looks like we accidentally get something larger than xmax sometimes, this should be kosher
        vals = vals[0:-2]
        x_axis = x_axis[0:-2]
        if plot == True:
            plt.plot(x_axis, vals, label=field)

        result = ', '.join(map(lambda v: np.format_float_positional(v, precision=precision), vals))
        x = ', '.join(map(lambda v: np.format_float_positional(v, precision=precision), x_axis))
        self.math_funcs[field] = { "x" : x, "dens" : result}
        return x, result
    
    def get_upstream_value(self, field):
        vals = self._get_field_values(field)
        return np.mean(vals[-10:-1])




def main():
    # Example usage:
    yt.enable_plugins()
    data_path = "/pscratch/sd/d/dschnei/FLASH_3D_noshield/MagShockZ_hdf5_plt_cnt_0009"
    ds = yt.load_for_osiris(data_path)
    start_point = (0, 0.01, 0)
    end_point = (0, 0.3, 0)
    lineout = Ray(ds, start_point, end_point)
    # lineout.show_ray("magx")
    # lineout.show_lineout("magx")

    fit_result = lineout.fit("sidens", degree=8, fit_func="piecewise", plot=True)
    print(fit_result)

if __name__ == "__main__":
    #TODO Fix this
    # import sys
    # parser = sys.argv(
    #                     prog='fitting_functions.py',
    #                     description='',
    #                     epilog='Text at the bottom of help')
    main()
