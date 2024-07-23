import re

def parse_sections(input_file):
    section_name_pattern = re.compile(r'^([^!{}\s]+)$')
    section_start_pattern = re.compile(r'^\s*{\s*$')
    section_end_pattern = re.compile(r'^\s*}\s*$')
    assignment_pattern = re.compile(r'([^=]+?)\s*=\s*(.+)')

    sections = {}
    section_stack = []  # Stack to track current section and nesting
    current_section = None
    species_name = None  # Track the name of the species

    def try_convert_to_float(val):
        try:
            return float(val)
        except ValueError:
            return val  # Return the original value if conversion fails

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip().split('!', 1)[0]  # Remove comments and whitespace

            if section_start_pattern.match(line):
                if current_section:
                    section_stack.append(current_section)
                continue

            if section_end_pattern.match(line):
                if section_stack:
                    current_section = section_stack.pop()  # Go back to the previous section
                else:
                    current_section = None  # No more sections to pop, reset current_section
                continue

            name_match = section_name_pattern.match(line)
            if name_match:
                current_section = name_match.group(1)
                if current_section.lower() == "species":
                    species_name = None  # Reset species name for a new species section
                continue

            assignment_match = assignment_pattern.match(line)
            if assignment_match:
                key, value = assignment_match.groups()
                print(key,value)
                key = key.strip().strip('"').rstrip(',')  # Strip whitespace, quotes, and trailing commas
                value = value.strip().strip('"').rstrip(',')

                if species_name is None and current_section.lower() == "species" and key.lower() == "name":
                    species_name = value.strip().rstrip(',').strip('"')
                    sections[species_name] = {}  # Initialize the species entry
                    continue

                # Determine the correct dictionary to update based on nesting
                target_dict = sections
                if species_name:
                    target_dict = sections[species_name]

                for sec in section_stack:
                    if sec not in target_dict:
                        target_dict[sec] = {}
                    target_dict = target_dict[sec]

                # Convert value to tuple, attempting to convert each element to a float
                value_as_tuple = tuple(try_convert_to_float(val) for val in value.split(','))
                target_dict[key] = value_as_tuple

    return sections

# Example usage
input_file_path = '/home/david/MagShockZ/input_files/SBS.1d'
sections = parse_sections(input_file_path)

print(sections.keys())
print(sections['el_mag_fld'].keys())
