import pandas as pd

# Load CSVs into Pandas DataFrames
hmi_df = pd.read_csv("raw-data/SHARP_hmi_data.csv")
cgem_df = pd.read_csv("raw-data/SHARP_cgem_data.csv")

# Perform an inner merge on NOAA_AR and T_REC
merged_df = pd.merge(hmi_df, cgem_df, on=["NOAA_AR", "T_REC", "HARPNUM"], how="inner")

# Clean the 'timestamp' column to remove any unwanted spaces and characters
merged_df['T_REC'] = merged_df['T_REC'].str.replace('_', ' ').str.replace('TAI', '').str.strip()
# Assuming the column containing the timestamp is called 'T_REC'
merged_df['T_REC'] = pd.to_datetime(merged_df['T_REC'].str.replace('_', ' ').str.replace('TAI', ''), format='%Y.%m.%d %H:%M:%S')

# Display the result
print(merged_df.shape[0])
print(merged_df)

merged_df.to_csv("raw-data/SHARP_hmi_cgem_data.csv")
print("Saved to raw-data/SHARP_hmi_cgem_data.csv")
