import re

# Path to the JSON file
input_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
output_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data_cleaned.json"

def clean_json(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove trailing commas before closing braces/brackets
    content = re.sub(r",\s*([\]}])", r"\1", content)
    
    # Ensure the JSON is wrapped in a list if multiple objects are present
    if not content.strip().startswith("["):
        content = f"[{content}]"

    # Write the cleaned content to a new file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Cleaned JSON saved to {output_path}")

if __name__ == "__main__":
    clean_json(input_file, output_file)