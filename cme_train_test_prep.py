import pandas as pd
from sklearn.model_selection import train_test_split

# Load merged SHARP data with flares, CMEs, and SEPs
raw_sharp_data = pd.read_csv("processed-data/SHARP_CME_SEP.csv")

# Create the 'Label' column based on the presence of 'linkedCME'
raw_sharp_data["Label"] = raw_sharp_data["linkedCMETime"].apply(lambda x: "P" if pd.notna(x) else "N")

# Define the columns to keep
columns_order = [
    "Label", "T_REC", "NOAA_AR", "HARPNUM", "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "AREA_ACR", "MEANPOT",
    "R_VALUE", "SHRGT45", "MEANGAM", "MEANJZH", "MEANGBT", "MEANGBZ", "MEANJZD", "MEANGBH", "MEANSHR", "MEANALP"
]

# Keep only the selected columns in the specified order
filtered_sharp_data = raw_sharp_data[columns_order].copy()

# Exclude the first four columns from normalization
columns_to_normalize = columns_order[4:]  # All columns except the first 4
# Apply normalization: (value - mean) / std deviation
filtered_sharp_data.loc[:, columns_to_normalize] = filtered_sharp_data.loc[:, columns_to_normalize].apply(lambda x: (x - x.mean()) / x.std())

# Split the data into train (70%) and test (30%)
train_data, test_data = train_test_split(filtered_sharp_data, test_size=0.3, random_state=42, shuffle=False)

# Save the split data into separate CSV files
train_data.to_csv("../RNN-CME-prediction/CME_data_samples/rnn_cme_train.csv", index=False)
test_data.to_csv("../RNN-CME-prediction/CME_data_samples/rnn_cme_test.csv", index=False)

train_data.to_csv("training-data/rnn_cme_train.csv", index=False)
test_data.to_csv("test-data/rnn_cme_test.csv", index=False)

print(train_data)
print("Training data saved to train-data/rnn_cme_train.csv")
print("Testing data saved to test-data/rnn_cme_test.csv")
