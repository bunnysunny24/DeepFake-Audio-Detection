import json

def truncate_json_file(input_file, output_file, error_position):
    """
    Truncate the JSON file at the specified error position and ensure it ends with a valid JSON structure.
    """
    try:
        # Read the file as plain text
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Truncate the content up to the error position
        truncated_content = content[:error_position]

        # Ensure the truncated content forms a valid JSON structure
        # Remove trailing commas and close the JSON array
        truncated_content = truncated_content.rstrip(", \n") + "]"

        # Validate the cleaned JSON
        data = json.loads(truncated_content)  # This will raise an error if the JSON is invalid
        print("JSON truncated and validated successfully.")

        # Write the cleaned JSON to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Cleaned JSON saved to {output_file}")

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError while truncating: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
    output_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data_v2.json"
    error_position = 137904848  # Character position of the error
    truncate_json_file(input_file, output_file, error_position)