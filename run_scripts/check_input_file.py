class CheckError(Exception):
    pass

def find_species_keys(d):
    nonspecies_keys = ['node_conf', 'grid', 'time_step', 'restart', 'space', 'time', 'emf_bound', 'diag_emf', 'particles'] # Add more keys as needed
    species_keys = []

    for key in d.keys():
        if key not in nonspecies_keys:
            species_keys.append(key)

    return species_keys

def check_dt_lessthan_dx(d):
    try:
        dx = (d['space']['xmax(1:1)'][0] - d['space']['xmin(1:1)'][0])/d['grid']['nx_p(1:1)'][0]
        dt = d['time_step']['dt'][0]
        if dt > dx:  # Example condition
            raise CheckError("dt is greater than dx.")
    except KeyError:
        raise CheckError("Missing 'grid' or 'nx_p(1:1)' in input.")
    print()
    pass

def check_phase_space_spatial_bounds(d):
    species_keys = find_species_keys(d)
    print(f'Bounds of box are: {d["space"]["xmin(1:1)"][0]} to {d["space"]["xmax(1:1)"][0]}')
    for species in species_keys:
        phase_space_bounds = d[species]['diag_species']['ps_xmax(1:1)'][0]
        print(f'Phase space bounds for {species}: {phase_space_bounds}')
        if phase_space_bounds <  d["space"]["xmax(1:1)"][0] - d["space"]["xmin(1:1)"][0]:
            raise CheckError(f"Phase space bounds for {species} are less than spatial bounds.")
    print()
    pass

def check_species_mass(d):
    species_keys = find_species_keys(d)
    for species in species_keys:
        mass = d[species]['species']['rqm'][0]
        print(f'Mass of {species}: {mass}')
    print()
    pass

# Main function to perform all checks
def perform_checks(d):
    errors = []
    check_functions = [check_dt_lessthan_dx, check_phase_space_spatial_bounds,check_species_mass]  # List of check functions

    for check_function in check_functions:
        try:
            check_function(d)
        except CheckError as e:
            errors.append(str(e))

    if errors:
        for error in errors:
            print("Error:", error)
        # Optionally, raise an exception after all checks are done
        raise CheckError("One or more checks failed.")

# Usage
from parse_input_file import parse_sections

sections = parse_sections('/home/david/MagShockZ/input_files/magshockz-v1.1.1d')
print(sections.keys())
print(sections['ionsPiston'])

try:
    perform_checks(sections)
    print("Completed successfully.")
except CheckError:
    print("Completed with errors.")