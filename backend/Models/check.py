import pandas as pd

# Load the Excel file, assuming no header
df = pd.read_excel('checking.xlsx', header=None)

# The first column is now column 0
first_column = df[0]

# File to check
file_to_check = 'dev/036900.mp4'

# Check if it exists
if file_to_check in first_column.values:
    print(f"{file_to_check} exists in the column!")
else:
    print(f"{file_to_check} does NOT exist in the column.")
