import re

def parse_sections(input_file):
    section_name_pattern = re.compile(r'^([^!{}\s]+)$')
    section_start_pattern = re.compile(r'^\s*{\s*$')
    section_end_pattern = re.compile(r'^\s*}\s*$')
    assignment_pattern = re.compile(r'([^=]+)=\s*(.+)')  # Pattern to match assignments
    
    sections = {}
    current_section = None
    current_species = None  # Track the current species name
    section_name_pending = False

    with open(input_file, 'r') as file:
        for line in file:
            line = line.split('!', 1)[0].strip()  # Remove comments and whitespace
            
            if section_name_pending:
                if section_start_pattern.match(line):
                    if current_section == "species":
                        # For species, we delay adding to sections until name is known
                        temp_section = {}
                    else:
                        sections[current_section] = {}
                    section_name_pending = False
                    continue
                else:
                    section_name_pending = False
                    current_section = None
            
            if current_section is not None:
                if section_end_pattern.match(line):
                    if current_section == "species" and current_species:
                        # Now that the species section is complete, add it under the species name
                        sections[current_species] = temp_section
                        current_species = None  # Reset current_species for the next species section
                    current_section = None
                else:
                    assignment_match = assignment_pattern.match(line)
                    if assignment_match:
                        key = assignment_match.group(1).strip()
                        value = assignment_match.group(2).strip()
                        values_list = [val.strip().strip('"') for val in value.split(',')]

                        # Attempt to convert each value to a float if possible
                        converted_values_list = []
                        for val in values_list:
                            try:
                                # Try converting to float
                                converted_values_list.append(float(val))
                            except ValueError:
                                # If conversion fails, keep the value as a string
                                converted_values_list.append(val)

                        if len(converted_values_list) == 1:
                            # If there's only one value, use it directly without a tuple
                            final_value = converted_values_list[0]
                        else:
                            if converted_values_list[-1] == '':
                                converted_values_list.pop()  # Remove the last element if it's an empty string
                            final_value = tuple(converted_values_list)

                        # Decide where to add the assignment based on current_species
                        target_section = temp_section if current_section == "species" else sections.get(current_species, sections[current_section])
                        target_section[key] = final_value
            else:
                name_match = section_name_pattern.match(line)
                if name_match:
                    current_section = name_match.group(1)
                    section_name_pending = True

    return sections

# Example usage
input_file_path = '/home/david/MagShockZ/input_files/magshockz-v1.1.1d'
sections = parse_sections(input_file_path)
print(sections)