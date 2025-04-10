import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create a new 'Label' column based on multiple conditions
# def assign_label(row):
#     flare_class = str(row["flareClass"]).strip()  # Ensure string and strip whitespaces
#     if flare_class.startswith(("M", "X")):
#         if pd.notna(row["linkedSEPTime"]):
#             return "P"  # Positive class if CME exists
#         else:
#             return "N"  # Negative class if CME does not exist
#     else:
#         return "padding"  # Retain rows where flareClass is neither M nor X

# Load merged SHARP data with flares, CMEs, and SEPs
raw_sharp_data = pd.read_csv("processed-data/SHARP_CME_SEP.csv", low_memory=False)

# filtered_sharp_data = raw_sharp_data.copy()
# # Convert flareClass to string to avoid AttributeError
# filtered_sharp_data["flareClass"] = filtered_sharp_data["flareClass"].astype(str)
# # Apply the condition to assign labels
# filtered_sharp_data["Label"] = filtered_sharp_data.apply(assign_label, axis=1)

# Create the 'Label' column based on the presence of 'linkedCME'
raw_sharp_data["Label"] = raw_sharp_data["linkedCMETime"].apply(lambda x: "P" if pd.notna(x) else "N")
raw_sharp_data["flareClass"] = raw_sharp_data["flareClass"].str.strip()
# Filter rows where 'flareClass' is either 'M' or 'X'
filtered_sharp_data = raw_sharp_data[raw_sharp_data["flareClass"].isin(["M", "X"])].copy()

# Define the columns to keep
columns_order = [
    "Label", "T_REC", "NOAA_AR", "HARPNUM", "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "AREA_ACR", "MEANPOT",
    "R_VALUE", "SHRGT45", "MEANGAM", "MEANJZH", "MEANGBT", "MEANGBZ", "MEANJZD", "MEANGBH", "MEANSHR", "MEANALP", "TOTBSQ"
]

# Keep only the selected columns in the specified order
filtered_sharp_data = filtered_sharp_data[columns_order]

# Print count before removing duplicates
print(f"Row count before filtering and removing duplicates: {len(filtered_sharp_data)}")
# Drop rows with any NaN values
filtered_sharp_data.dropna(inplace=True)
filtered_sharp_data.reset_index(drop=True, inplace=True)
filtered_sharp_data = filtered_sharp_data.drop_duplicates()

# if NOAA_AR has two flares associated keep the one that has a CME associated with it too
# Group by T_REC and filter to keep the row where Label is 'P' if there are duplicates
filtered_sharp_data = filtered_sharp_data.groupby('T_REC').apply(
    lambda x: x[x['Label'] == 'P'] if (x['Label'] == 'P').any() else x
).reset_index(drop=True)

# Exclude the first four columns from normalization
columns_to_normalize = columns_order[4:]  # All columns except the first 4
# Replace infinite values with NaN in the specified columns
filtered_sharp_data[columns_to_normalize] = filtered_sharp_data[columns_to_normalize].replace([np.inf, -np.inf], np.nan)

# Drop rows where any of the specified columns have NaN values (which were originally infinite)
filtered_sharp_data.dropna(subset=columns_to_normalize, inplace=True)
# Print count after removing duplicates
print(f"Row count after filtering and removing duplicates: {len(filtered_sharp_data)}")

# Apply normalization: (value - mean) / std deviation
filtered_sharp_data.loc[:, columns_to_normalize] = filtered_sharp_data.loc[:, columns_to_normalize].apply(lambda x: ((x - x.mean()) / x.std()).round(9))

# Convert HARPNUM to integer, coercing errors to NaN if they are not convertible
filtered_sharp_data.loc[:, 'HARPNUM'] = pd.to_numeric(filtered_sharp_data.loc[:, 'HARPNUM'], errors='coerce')
# Sort the data by both T_REC and HARPNUM
filtered_sharp_data = filtered_sharp_data.sort_values(by=['HARPNUM', 'T_REC'], ascending=[True, True], ignore_index=True)

# Split the data into train (80%) and test (20%) with equal distribution
train_pos, test_pos = train_test_split(filtered_sharp_data[filtered_sharp_data['Label'] == 'P'], test_size=0.2, shuffle=False)
train_neg, test_neg = train_test_split(filtered_sharp_data[filtered_sharp_data['Label'] == 'N'], test_size=0.2, shuffle=False)

train_data = pd.concat([train_pos, train_neg]).sort_index()
test_data = pd.concat([test_pos, test_neg]).sort_index()

# Save the split data into separate CSV files
train_data.to_csv("../SEP-prediction/SEP_Package/data/bilstm_sep_train.csv", index=False)
test_data.to_csv("../SEP-prediction/SEP_Package/data/bilstm_sep_test.csv", index=False)

train_data.to_csv("train-data/bilstm_sep_train.csv", index=False)
test_data.to_csv("test-data/bilstm_sep_test.csv", index=False)

print("Training data saved to train-data/bilstm_sep_train.csv")
print("Testing data saved to test-data/bilstm_sep_test.csv")
