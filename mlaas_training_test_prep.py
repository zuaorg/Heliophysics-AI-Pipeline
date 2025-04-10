import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load merged SHARP data with flares, CMEs, and SEPs
raw_sharp_data = pd.read_csv("processed-data/SHARP_CME_SEP.csv")

# Define the columns to keep
columns_to_keep = [
    "flareClass", "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH",
    "SAVNCPP", "USFLUX", "AREA_ACR", "TOTFZ", "MEANPOT", "R_VALUE", "EPSZ", "SHRGT45"
]

# Keep only the selected columns
filtered_sharp_data = raw_sharp_data[columns_to_keep].copy()

# Drop rows with any NaN values
filtered_sharp_data.dropna(inplace=True)
filtered_sharp_data.reset_index(drop=True, inplace=True)

# # Handle NaNs first
# filtered_sharp_data.loc[:, "flareClass"] = filtered_sharp_data["flareClass"].fillna("N/A")
# filtered_sharp_data.loc[:, "flareClass"] = filtered_sharp_data["flareClass"].astype(str)
# # Fill NaN values with 0
# filtered_sharp_data = filtered_sharp_data.fillna(0)
# Select only numeric columns
numeric_data = filtered_sharp_data.select_dtypes(include=[np.number])
# Find rows with inf or -inf values
rows_with_inf = numeric_data.isin([np.inf, -np.inf]).any(axis=1)
# Drop these rows
filtered_sharp_data = filtered_sharp_data[~rows_with_inf].reset_index(drop=True)
filtered_sharp_data = filtered_sharp_data.drop_duplicates()

# Split the data into train (80%) and test (20%)
train_data, test_data = train_test_split(filtered_sharp_data, test_size=0.2, random_state=42)

# Save the split data into separate CSV files
train_data.to_csv("../Machine-learning-as-a-service/data/train_data/mlaas_train.csv", index=False)
test_data.to_csv("../Machine-learning-as-a-service/data/test_data/mlaas_test.csv", index=False)

train_data.to_csv("train-data/mlaas_train.csv", index=False)
test_data.to_csv("test-data/mlaas_test.csv", index=False)

print(train_data)
print("Training data saved to train-data/mlaas_train.csv")
print("Testing data saved to test-data/mlaas_test.csv")
