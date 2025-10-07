import numpy as np
import matplotlib.pyplot as plt
import yt
import pwlf


class Ray:
    """
    Class to handle the lineouts and fitting from FLASH to OSIRIS.
    """
    def __init__(self, ds, start_pt, end_pt, reference_density = 5e18, rqm_factor = 50):
        if isinstance(ds, str):
            yt.enable_plugins()
            self.ds = yt.load_for_osiris(ds, rqm_factor=rqm_factor)
        elif isinstance(ds, yt.frontends.flash.data_structures.FLASHDataset):
            self.ds = ds
        self.rqm_factor = rqm_factor 

        omega_pe = np.sqrt(4 * np.pi * yt.units.elementary_charge_cgs**2 * reference_density / yt.units.electron_mass_cgs) # rad/s, assuming electron density of 1e18 cm^-3

        B_norm = (omega_pe * yt.units.electron_mass_cgs * yt.units.speed_of_light_cgs) / yt.units.elementary_charge_cgs
        E_norm = B_norm * yt.units.speed_of_light_cgs / np.sqrt(self.rqm_factor)
        v_norm = yt.units.speed_of_light_cgs / np.sqrt(self.rqm_factor)
        vth_ele_norm = np.sqrt(yt.units.electron_mass_cgs * yt.units.speed_of_light_cgs**2 / yt.units.boltzmann_constant_cgs)
        vth_ion_norm = np.sqrt(yt.units.electron_mass_cgs * yt.units.speed_of_light_cgs**2 / yt.units.boltzmann_constant_cgs) * np.sqrt(3800 / self.rqm_factor) # This is using a very rough calculation for the rqm of silicon and Aluminum. Will need to change if you are using more complicated species


        self.start_pt = start_pt
        self.end_pt = end_pt
        self.length = self._get_distance_axis()
        self.osiris_length = self.length / (yt.units.speed_of_light_cgs / omega_pe)  # Normalize to OSIRIS units
        self.osiris_length = (self.osiris_length - self.osiris_length[0]).value  # Center the x values to start at 0
        self.math_funcs = {}


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
            'tele': vth_ele_norm,
            # Thermal velocity for ions
            'tion': vth_ion_norm,
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

        # thermal velocities to temperatures are the only nonlinear fields, so they need their own treatment where we take sqrt
        if field == "tele" or field == "tion":
            # vthe = sqrt(T/mc^2)
            # vthi = sqrt(T/mc^2) * 1 / sqrt(rqm)
            values = (np.sqrt(ray[(field)][ray_sort]) / self.normalizations[field]).value
        else:
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


    def fit(self, field, degree=3, left_value=None, right_value=None, fit_func = "polynomial", plot=False):
        """
        Fit a polynomial to the lineout data.
        """
        if field.endswith("dens"):
            print("suggest using fit_density for initializing density fields")

        vals = self._get_field_values(field)
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
        precision = 5
        result = ''

        if fit_func == "polynomial":
            # In this case, the function can be defined in all space
            if left_value is None and right_value is None:
                for i in range(len(coefficients)):
                    result = f"{result}({np.format_float_scientific(coefficients[i], sign=False)})*(x2)^("+str(degree-i) + ") +"
                result = f"{result.strip(' +')}"

            # In this case, the function is only defined in the region of interest
            if left_value is not None and right_value is not None:
                result = f"{result}if(x2 < {round(self.osiris_length[0],precision)}, {left_value}, if(x2 < {round(self.osiris_length[-1], precision)}, "
                for i in range(len(coefficients)):
                    result = f"{result} ({np.format_float_scientific(coefficients[i], precision=precision, sign=False)})*(x2^("+str(degree-i) + ") +"
                result = result.strip(' +')+f", {right_value}))"
            # In this case, the function starts at the left side of the box and goes to the right boundary
            if left_value is None and right_value is not None:
                result = f"{result}if(x2 < {round(self.osiris_length[-1],precision)}, "
                for i in range(len(coefficients)):
                    result = f"{result} ({np.format_float_scientific(coefficients[i], precision=precision, sign=False)})*(x2^("+str(degree-i) + ") +"
                result = result.strip(' +')+f", {right_value})"
        if fit_func == "exponential":
            if left_value is None and right_value is None:
                result = f"{result}({np.format_float_scientific(coefficients[0], precision=precision, sign=False)})*exp(x2 + {np.format_float_scientific(coefficients[1], precision=precision, sign=False)})"

            # In this case, the function is only defined in the region of interest
            if left_value is not None and right_value is not None:
                result = f"{result}if(x2 < {round(self.osiris_length[0],precision)}, {left_value}, if(x2 < {round(self.osiris_length[-1], precision)}, "
                result = f"{result} ({np.format_float_scientific(coefficients[0], precision=precision, sign=False)})*exp(x2 + {np.format_float_scientific(coefficients[1], precision=precision, sign=False)})"

            # In this case, the function starts at the left side of the box and goes to the right boundary
            if left_value is None and right_value is not None:
                result = f"{result}if(x2 < {round(self.osiris_length[-1],precision)}, "
                result = f"{result} ({np.format_float_scientific(coefficients[0], precision=precision, sign=False)})*exp(x2 + {np.format_float_scientific(coefficients[1], precision=precision, sign=False)})"

        if fit_func == "piecewise":
            extra_parentheses = 0
            if left_value is not None:
                result += f"if(x2 < {round(pwlf_model.fit_breaks[0], precision)}, {left_value}, "
                extra_parentheses += 1

            for i in range(len(pwlf_model.fit_breaks)-1):
                result += f"if(x2 < {np.format_float_scientific(pwlf_model.fit_breaks[i+1],precision)}, "
                result += f"x2*({np.format_float_scientific(pwlf_model.slopes[i], precision)}) + ({np.format_float_scientific(pwlf_model.intercepts[i], precision)}), "

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




def main():
    # Example usage:
    yt.enable_plugins()
    data_path = "/mnt/cellar/shared/simulations/FLASH_MagShockZ3D-Trantham_06-2024/MAGON/MagShockZ_hdf5_chk_0005"
    ds = yt.load_for_osiris(data_path)
    start_point = (0, 0.01, 0)
    end_point = (0, 0.3, 0)
    lineout = Ray(ds, start_point, end_point)
    # lineout.show_ray("magx")
    # lineout.show_lineout("magx")

    fit_result = lineout.fit("sidens", degree=8, fit_func="piecewise", plot=True)
    print(fit_result)

if __name__ == "__main__":
    import sys
    parser = sys.argparse.ArgumentParser(
                        prog='fitting_functions.py',
                        description='',
                        epilog='Text at the bottom of help')
    main()
