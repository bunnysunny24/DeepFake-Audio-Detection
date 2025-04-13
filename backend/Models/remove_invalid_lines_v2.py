import json

def is_valid_json_line(line):
    """
    Check if a line is part of a valid JSON structure.
    """
    try:
        # Attempt to load the line as JSON
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False

def remove_invalid_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    inside_json = False
    for line in lines:
        # Check for the start of JSON structure
        if line.strip().startswith('{') or line.strip().startswith('['):
            inside_json = True

        # Skip invalid lines that don't belong to JSON structure
        if not inside_json and not line.strip().startswith(('}', ']')):
            continue

        # Add valid JSON lines
        if is_valid_json_line(line.strip()) or line.strip().startswith(('}', ']')):
            cleaned_lines.append(line)
        else:
            print(f"Removed invalid line: {line.strip()}")

    # Write the cleaned content to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned JSON saved to {output_file}")

if __name__ == "__main__":
    input_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data_final.json"
    output_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data_corrected.json"
    remove_invalid_lines(input_file, output_file)