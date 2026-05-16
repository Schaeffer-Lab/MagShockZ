"""
Utility to parse and plot OSIRIS math function strings
This helps validate that the generated functions are correct
"""

import numpy as np
import matplotlib.pyplot as plt
import re

def parse_osiris_function(func_str, dim_var='x1'):
    """
    Parse an OSIRIS math function string and convert it to a Python function
    
    Parameters:
    -----------
    func_str : str
        OSIRIS math function string (e.g., piecewise if() statements)
    dim_var : str
        The dimension variable used ('x1', 'x2', or 'x3')
    
    Returns:
    --------
    callable
        Python function that can be evaluated
    """
    # Replace OSIRIS syntax with Python syntax
    python_str = func_str.replace(dim_var, 'x')
    
    # Handle if statements - OSIRIS uses if(condition, true_val, false_val)
    # We need to convert to Python's (true_val if condition else false_val)
    def convert_if(match_obj):
        # This is a simple conversion that handles nested if statements
        content = match_obj.group(1)
        # Find the condition (everything before first comma)
        parts = []
        depth = 0
        current = ''
        for char in content:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(current.strip())
                current = ''
                continue
            current += char
        parts.append(current.strip())
        
        if len(parts) != 3:
            return match_obj.group(0)  # Return original if can't parse
        
        condition, true_val, false_val = parts
        return f"({true_val} if {condition} else {false_val})"
    
    # Recursively replace if statements from innermost to outermost
    max_iterations = 100
    iteration = 0
    while 'if(' in python_str and iteration < max_iterations:
        # Find innermost if statement
        python_str = re.sub(r'if\(((?:[^()]+|\([^()]*\))*)\)', convert_if, python_str)
        iteration += 1
    
    # Create a lambda function that can be evaluated
    try:
        # Safe evaluation with numpy functions available
        func = lambda x: eval(python_str, {"__builtins__": {}}, 
                             {"x": x, "np": np, "exp": np.exp})
        return func
    except Exception as e:
        print(f"Error creating function: {e}")
        print(f"String: {python_str}")
        return None

def plot_osiris_function(func_str, x_min, x_max, dim_var='x1', num_points=1000, 
                         title=None, ax=None, label=None):
    """
    Plot an OSIRIS math function
    
    Parameters:
    -----------
    func_str : str
        OSIRIS math function string
    x_min, x_max : float
        Range to plot
    dim_var : str
        Dimension variable used in the function
    num_points : int
        Number of points to evaluate
    title : str, optional
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on (creates new figure if None)
    label : str, optional
        Label for the plot line
    """
    func = parse_osiris_function(func_str, dim_var)
    
    if func is None:
        print("Failed to parse function")
        return None
    
    x = np.linspace(x_min, x_max, num_points)
    
    try:
        # Vectorize the function
        y = np.array([func(xi) for xi in x])
    except Exception as e:
        print(f"Error evaluating function: {e}")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, y, label=label if label else 'OSIRIS function')
    ax.set_xlabel(f'{dim_var} [code units]')
    ax.set_ylabel('Value [code units]')
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax

def validate_input_file(input_file_path, dim_var='x1'):
    """
    Read an OSIRIS input file and plot all the math functions found in it
    
    Parameters:
    -----------
    input_file_path : str
        Path to the OSIRIS input file
    dim_var : str
        Dimension variable to expect
    """
    with open(input_file_path, 'r') as f:
        content = f.read()
    
    # Find all spatial_uth and spatial_ufl definitions
    pattern = rf'spatial_u[tf]l?\(\d+\)\s*=\s*"([^"]+)"'
    matches = re.findall(pattern, content)
    
    # Also find init_b_mfunc and init_e_mfunc
    pattern2 = rf'init_[be]_mfunc\(\d+\)\s*=\s*"([^"]+)"'
    matches2 = re.findall(pattern2, content)
    
    all_funcs = matches + matches2
    
    # Get x range from space section
    xmin_match = re.search(r'xmin\(1:1\)\s*=\s*([-\d.e+]+)', content)
    xmax_match = re.search(r'xmax\(1:1\)\s*=\s*([-\d.e+]+)', content)
    
    if xmin_match and xmax_match:
        xmin = float(xmin_match.group(1))
        xmax = float(xmax_match.group(1))
    else:
        print("Could not find xmin/xmax in input file")
        xmin, xmax = 0, 1000
    
    print(f"Found {len(all_funcs)} math functions in input file")
    print(f"Spatial range: {xmin} to {xmax}")
    
    if all_funcs:
        # Create a grid of subplots
        n_funcs = len(all_funcs)
        ncols = min(3, n_funcs)
        nrows = (n_funcs + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        if n_funcs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 else axes
        
        for i, func_str in enumerate(all_funcs):
            if i < len(axes):
                plot_osiris_function(func_str, xmin, xmax, dim_var=dim_var, 
                                   ax=axes[i], title=f'Function {i+1}')
        
        # Hide unused subplots
        for i in range(len(all_funcs), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(input_file_path.replace('.txt', '_functions_plot.png'), dpi=150)
        print(f"Saved plot to {input_file_path.replace('.txt', '_functions_plot.png')}")
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot OSIRIS math functions from input file")
    parser.add_argument('input_file', type=str, help="Path to OSIRIS input file")
    parser.add_argument('--dim_var', type=str, default='x1', 
                       help="Dimension variable used (x1, x2, or x3)")
    
    args = parser.parse_args()
    
    validate_input_file(args.input_file, dim_var=args.dim_var)
