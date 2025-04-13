import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create a new 'Label' column based on multiple conditions
# def assign_label(row):
#     flare_class = str(row["flareClass"]).strip()  # Ensure string and strip whitespaces
#     if flare_class.startswith(("M", "X")):
#         if pd.notna(row["linkedCMETime"]):
#             return "P"  # Positive class if CME exists
#         else:
#             return "N"  # Negative class if CME does not exist
#     else:
#         return "padding"  # Retain rows where flareClass is neither M nor X

# Load merged SHARP data with flares, CMEs, and SEPs
raw_sharp_data = pd.read_csv("../processed-data/SHARP_CME_SEP.csv")
# Print count before removing duplicates
print(f"Row count before filtering and removing duplicates: {len(raw_sharp_data)}")
# Add an empty column "Label" with None as the initial value for all rows
raw_sharp_data["CMELabel"] = None
raw_sharp_data["SEPLabel"] = None

# Define the columns to keep
columns_order = [
    "flareClass", "CMELabel", "SEPLabel", "T_REC", "NOAA_AR", "HARPNUM", "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "AREA_ACR", "MEANPOT",
    "R_VALUE", "SHRGT45", "MEANGAM", "MEANJZH", "MEANGBT", "MEANGBZ", "MEANJZD", "MEANGBH", "MEANSHR", "MEANALP", "TOTBSQ", "linkedCMETime", "linkedSEPTime"
]
# Keep only the selected columns in the specified order
filtered_sharp_data = raw_sharp_data[columns_order].copy()
filtered_sharp_data = filtered_sharp_data.replace([np.inf, -np.inf], 0)
filtered_sharp_data = filtered_sharp_data.drop_duplicates()

# Convert HARPNUM to integer
filtered_sharp_data.loc[:, 'HARPNUM'] = pd.to_numeric(filtered_sharp_data.loc[:, 'HARPNUM'], errors='coerce')
# # Identify rows where 'flareClass' is NaN
# na_rows = filtered_sharp_data[filtered_sharp_data['flareClass'].isna()]
# # Randomly sample 10% of the NaN rows
# sampled_na_rows = na_rows.sample(frac=0.015, random_state=42)
# # Get all rows that already have a valid 'flareClass'
# non_na_rows = filtered_sharp_data[~filtered_sharp_data['flareClass'].isna()]
# # Combine the non-NaN rows with the 10% sampled NaN rows
# filtered_sharp_data = pd.concat([non_na_rows, sampled_na_rows], axis=0)
# Replace NaN values in 'flareClass' with the string 'N/A'
# filtered_sharp_data = filtered_sharp_data.dropna(subset=['flareClass'])
# Optionally, reset the index
# filtered_sharp_data.reset_index(drop=True, inplace=True)
filtered_sharp_data = filtered_sharp_data.dropna(subset=['flareClass'])

train_data, test_data = train_test_split(filtered_sharp_data, test_size=0.25, shuffle=False)

flare_sharp_data = train_data.copy()
# Filter rows where 'flareClass' is either 'M' or 'X'
train_data = train_data[train_data["flareClass"].isin(["M", "X"])]


# Create the 'Label' column based on the presence of 'linkedCME'
cme_sharp_data = train_data.copy()
cme_sharp_data["CMELabel"] = cme_sharp_data["linkedCMETime"].apply(lambda x: "P" if pd.notna(x) else "N")
test_data["CMELabel"] = test_data["linkedCMETime"].apply(lambda x: "P" if pd.notna(x) else "N")
# Create the 'Label' column based on the presence of 'linkedSEP'
sep_sharp_data = train_data.copy()
sep_sharp_data["SEPLabel"] = sep_sharp_data["linkedSEPTime"].apply(lambda x: "P" if pd.notna(x) else "N")
test_data["SEPLabel"] = test_data["linkedSEPTime"].apply(lambda x: "P" if pd.notna(x) else "N")
test_data = test_data.drop(columns=['linkedSEPTime', 'linkedCMETime'])
cme_sharp_data = cme_sharp_data.drop(columns=['flareClass', 'SEPLabel', 'linkedCMETime', 'linkedSEPTime'])
sep_sharp_data = sep_sharp_data.drop(columns=['flareClass', 'CMELabel', 'linkedSEPTime', 'linkedCMETime'])

test_data = test_data.fillna(0)
flare_sharp_data = flare_sharp_data.fillna(0)
cme_sharp_data = cme_sharp_data.fillna(0)
sep_sharp_data = sep_sharp_data.fillna(0)
test_data = test_data.drop_duplicates()
flare_sharp_data = flare_sharp_data.drop_duplicates()
cme_sharp_data = cme_sharp_data.drop_duplicates()
sep_sharp_data = sep_sharp_data.drop_duplicates()

# if NOAA_AR has two flares associated keep the one that has a CME/SEP associated with it too
# Group by T_REC and filter to keep the row where Label is 'P' if there are duplicates
cme_sharp_data = cme_sharp_data.groupby('T_REC').apply(
    lambda x: x[x['CMELabel'] == 'P'] if (x['CMELabel'] == 'P').any() else x
).reset_index(drop=True)
sep_sharp_data = sep_sharp_data.groupby('T_REC').apply(
    lambda x: x[x['SEPLabel'] == 'P'] if (x['SEPLabel'] == 'P').any() else x
).reset_index(drop=True)

# Normalize 18 SHARP parameters
columns_to_normalize = columns_order[6:24]

# Print count after removing duplicates
print(f"Flare Row count after filtering and removing duplicates: {len(flare_sharp_data)}")
print(f"CME Row count after filtering and removing duplicates: {len(cme_sharp_data)}")
print(f"SEP Row count after filtering and removing duplicates: {len(sep_sharp_data)}")
print(f"Test Row count after filtering and removing duplicates: {len(test_data)}")

# Apply normalization: (value - mean) / std deviation
# cme_sharp_data.loc[:, columns_to_normalize] = cme_sharp_data.loc[:, columns_to_normalize].apply(lambda x: ((x - x.mean()) / x.std()).round(9))
# sep_sharp_data.loc[:, columns_to_normalize] = sep_sharp_data.loc[:, columns_to_normalize].apply(lambda x: ((x - x.mean()) / x.std()).round(9))

# Sort the data by both T_REC and HARPNUM
flare_sharp_data = flare_sharp_data.sort_values(by=['HARPNUM', 'T_REC'], ascending=[True, True], ignore_index=True)
cme_sharp_data = cme_sharp_data.sort_values(by=['HARPNUM', 'CMELabel', 'T_REC'], ascending=[True, True, True], ignore_index=True)
sep_sharp_data = sep_sharp_data.sort_values(by=['HARPNUM', 'SEPLabel', 'T_REC'], ascending=[True, True, True], ignore_index=True)
test_data = test_data.sort_values(by=['HARPNUM', 'CMELabel', 'SEPLabel', 'T_REC'], ascending=[True, True, True, True], ignore_index=True)

# Save the split data into separate CSV files
flare_sharp_data.to_csv("../train-data/flare_train.csv", index=False)
cme_sharp_data.to_csv("../train-data/cme_train.csv", index=False)
sep_sharp_data.to_csv("../train-data/sep_train.csv", index=False)
test_data.to_csv("../test-data/test_data.csv", index=False)

print("Training data saved to directory 'train-data'")
print("Testing data saved to directory 'test-data'")
