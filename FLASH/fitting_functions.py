import numpy as np
import matplotlib.pyplot as plt

def fit_to_region(start_pt: list, end_pt:list, field: str, ds, degree: int, precision: int, normalization:float, osiris, left_value = None, right_value = None):

    def get_distance_axis(start_pt, end_pt, ds):
        '''
        Helper function to get a distance axis for the ray on which you are taking a lineout
        '''
        euclidean_length_of_ray = np.sqrt((end_pt[0]-start_pt[0])**2 + (end_pt[1]-start_pt[1])**2 + (end_pt[2]-start_pt[2])**2)
        dist_from_origin = np.sqrt(start_pt[0]**2 + start_pt[1]**2 + start_pt[2]**2)
        return np.array(ds.ray(start_pt, end_pt)['t']*euclidean_length_of_ray+dist_from_origin)
    

    target_wall = 0.01 # This is hard coded based on the exact simulation I am working with, delete this line before using
    x = get_distance_axis(start_pt, end_pt, ds)-target_wall

    # normalize distances to simulation units
    x = x/osiris['length']
    
    # in order to make polynomials that aren't absolute garbage, we need to normalize the x values. This will make x span from 0 to 1 in the region of interest
    x_norm = (x - x[0])/(x[-1]-x[0])

    def get_field_values(start_pt, end_pt, field, ds, normalization):
        '''
        Helper function to get the field values for the ray on which you are taking a lineout
        '''
        ray = ds.ray(start_pt, end_pt)
        ray_sort = np.argsort(ray["t"])
        values = np.array(ray[('flash',field)][ray_sort])/normalization
        return values
    
    vals = get_field_values(start_pt, end_pt, field, ds, normalization)

    coefficients = np.polyfit(x_norm, vals, degree)

    # Create a polynomial function from the coefficients
    polynomial_function = np.poly1d(coefficients)

    # Calculate y values for the plotting range using the polynomial function
    x_norm_smooth = np.linspace(x_norm[0], x_norm[-1], 3000)
    y_fit = polynomial_function(x_norm_smooth)

    # Plot the original data points
    plt.scatter(x_norm*(x[-1]-x[0])+x[0], vals, label='Data Points')

    # Plot the polynomial fit
    plt.plot(x_norm_smooth*(x[-1]-x[0])+x[0], y_fit, color='red', label=f'{degree} Degree Polynomial Fit')

    plt.xlabel('dist [osiris_units]')
    plt.ylabel(f'{field} [osiris_units]')
    # pretty_plot()

    plt.legend()

    plt.show()

    # OSIRIS FORMATTING
    result = '\"'
    # In this case, the function can be defined in all space
    if left_value is None and right_value is None:
        # result = f"if(x1 < {round(x[0],precision)}, {left_value}, if("
        for i in range(len(coefficients)):
            result = f"{result}({np.format_float_positional(coefficients[i], precision=precision, sign=False)})*((x1 - {round(x[0],precision)})/{round(x[-1]-x[0],precision)})^("+str(degree-i) + ") +"
        print(f"{result.strip(' +')}\",")

    # In this case, the function is only defined in the region of interest
    if left_value is not None and right_value is not None:
        result = f"{result}if(x1 < {round(x[0],precision)}, {left_value}, if(x1 < {round(x[-1],precision)}, "
        for i in range(len(coefficients)):
            result = f"{result} ({np.format_float_positional(coefficients[i], precision=precision, sign=False)})*((x1 - {round(x[0],precision)})/{round(x[-1]-x[0],precision)})^("+str(degree-i) + ") +"
        print(result.strip(' +')+f", {right_value}))\",")
    # In this case, the function starts at the left side of the box and goes to the right boundary
    if left_value is None and right_value is not None:
        result = f"{result}if(x1 < {round(x[-1],precision)}, "
        for i in range(len(coefficients)):
            result = f"{result} ({np.format_float_positional(coefficients[i], precision=precision, sign=False)})*((x1 - {round(x[0],precision)})/{round(x[-1]-x[0],precision)})^("+str(degree-i) + ") +"
        print(result.strip(' +')+f", {right_value})\",")

    # DESMOS FORMATTING

    print("\n" + '-'*10 + "Desmos formatting to check your work" + '-'*10 + "\n")
    result = ''
    for i in range(len(coefficients)):
        result = f"{result} ({np.format_float_positional(coefficients[i],precision)})*((x - {round(x[0],precision)})/{round(x[-1]-x[0],precision)})^"+str(degree-i) + " +"
    print(result.strip(' +'))
    