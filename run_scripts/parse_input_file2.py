def parse_input_file(file_path):
    data = {}
    current_section = None
    inside_section = False
    seen_species = False  # Flag to indicate if the current section is a species section

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        # Check if we are at the start of a new section
        if not inside_section and i + 1 < len(lines) and lines[i + 1].strip() == '{':
            current_section = line
            if current_section == "species":
                seen_species = True  # Set flag if we should use the different naming convention
            if seen_species:    
                if current_section == "species":
                    for j in range(i + 2, len(lines)):
                        if lines[j].startswith('name'):
                            key, value = lines[j].split('=', 1)
                            assert key.strip() == 'name'
                            species_name = value.strip().strip('"').rstrip(',')
                            break
                data[species_name][current_section] = {}
            else:
                data[current_section] = {}
            inside_section = True
            continue  # Skip to the next iteration to avoid processing the '{' line

        # Check if we've reached the end of a section
        if inside_section and line == '}':
            inside_section = False
            continue

        # Process variable assignments within a section
        if inside_section and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().rstrip(',')
            # Remove comments from value if present
            value = value.split('!', 1)[0].strip()
            if seen_species:
                data[species_name][current_section][key] = value
            else:
                data[current_section][key] = value

    return data

# Example usage
file_path = '/home/david/MagShockZ/input_files/magshockz-v1.1.1d'
parsed_data = parse_input_file(file_path)
print(parsed_data.keys())