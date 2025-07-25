import pandas as pd
import os

file_path = os.path.join('..', 'src', 'furgoni.xlsx')
df_dict = pd.read_excel(file_path, sheet_name=None)

for sheet_name, df in df_dict.items():
    print(f"=== {sheet_name} ===")
    print("Columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: '{col}'")
    print()
