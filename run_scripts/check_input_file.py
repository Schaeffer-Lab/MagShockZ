import re

def check_variables(input_file):
    # Pattern to identify variable assignments, including multiple assignments separated by commas
    var_assignment_pattern = re.compile(r'(\w+)(?:\(([\d:]+)\))?\s*=\s*(.*?)(?:(?:,|$)\s*)')

    variables_dict = {}

    with open(input_file, 'r') as file:
        for line in file:
            # Strip comments: keep the part of the line before the first '!'
            line = line.split('!', 1)[0]
            # Find all matches of variable assignments in the line
            matches = var_assignment_pattern.findall(line)
            for match in matches:
                var_name, var_range, var_value = match
                # Handle optional range in the key
                key = f"{var_name}({var_range})" if var_range else var_name
                variables_dict[var_name] = var_value.strip()

    return variables_dict

# Example usage
input_file_path = '/home/david/MagShockZ/input_files/magshockz-v1.1.1d'
variables = check_variables(input_file_path)
print(variables)
print(f'{variables["dt"]} < {(float(variables["xmax"])-float(variables["xmin"]))/float(variables["nx_p"])}')