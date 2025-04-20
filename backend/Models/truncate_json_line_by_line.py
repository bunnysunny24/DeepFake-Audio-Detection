import json

def truncate_json_line_by_line(input_file, output_file, error_line):
    """
    Truncate the JSON file starting from the specified error line and ensure it ends with a valid JSON structure.
    """
    try:
        # Read the file line by line
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Retain only lines up to the problematic line
        valid_lines = lines[:error_line - 1]  # Exclude the problematic line

        # Ensure the JSON array is closed
        if valid_lines[-1].strip().endswith(","):
            valid_lines[-1] = valid_lines[-1].rstrip(", \n") + "\n"  # Remove trailing comma
        valid_lines.append("]\n")  # Close the JSON array

        # Join the valid lines into a single string
        cleaned_content = "".join(valid_lines)

        # Validate the cleaned JSON
        data = json.loads(cleaned_content)  # This will raise an error if the JSON is invalid
        print("JSON truncated and validated successfully.")

        # Write the cleaned JSON to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
        print(f"Cleaned JSON saved to {output_file}")

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError while truncating: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
    output_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data_cleaned_v2.json"
    error_line = 4891411  # Line number of the error
    truncate_json_line_by_line(input_file, output_file, error_line)