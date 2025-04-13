import json

def debug_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e.msg}")
            print(f"Line: {e.lineno}, Column: {e.colno}, Char position: {e.pos}")
            print("Inspect the file at this location for issues.")

if __name__ == "__main__":
    file_path = r"D:\Bunny\Deepfake\backend\combined_data\processed_data_fixed.json"
    debug_json(file_path)