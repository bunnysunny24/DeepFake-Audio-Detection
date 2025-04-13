import json

def clean_json_file(input_file, output_file, error_position):
    """
    Removes problematic data from the JSON file starting at the given error position.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Keep only the valid part of the JSON up to the error position
        cleaned_content = content[:error_position]

        # Try to ensure the JSON ends properly (close any open brackets)
        if cleaned_content.strip().endswith(","):
            cleaned_content = cleaned_content.strip()[:-1]  # Remove trailing comma
        cleaned_content += "]"  # Close the JSON array

        # Validate the cleaned JSON
        data = json.loads(cleaned_content)  # This will raise an error if the JSON is invalid
        print("JSON cleaned and validated successfully.")

        # Write the cleaned JSON to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Cleaned JSON saved to {output_file}")

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError while cleaning: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
    output_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data_cleaned.json"
    error_position = 137904848  # Character position of the error
    clean_json_file(input_file, output_file, error_position)