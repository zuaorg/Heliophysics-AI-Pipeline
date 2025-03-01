import pandas as pd

# Load merged SHARP and NOAA data
sharp_noaa_df = pd.read_csv("raw-data/SHARP_hmi_cgem_data.csv")

# Step 2: Load merged DONKI flare data
flare_df = pd.read_csv("raw-data/flare_data_with_CME_SEP.csv", on_bad_lines="skip")

# Filter rows where linkedEventType is 'CME'
cme_df = flare_df[flare_df["linkedEventType"] == "CME"].copy()
# Rename the column linkedEventType to linkedCME
cme_df = cme_df.rename(columns={"flrID": "flrIDCme", "linkedEventType": "linkedCME", "linkedEventTime": "linkedCMETime"})
print("CME count:", cme_df.shape[0])  # shape[0] gives the row count


# Filter rows where linkedEventType is 'SEP'
sep_df = flare_df[flare_df["linkedEventType"] == "SEP"].copy()
# Rename the column linkedEventType to linkedSEP
sep_df = sep_df.rename(columns={"flrID": "flrIDSep", "linkedEventType": "linkedSEP", "linkedEventTime": "linkedSEPTime"})
print("SEP count:", sep_df.shape[0])  # shape[0] gives the row count

# Convert NOAA_AR to string and strip whitespace
sharp_noaa_df["NOAA_AR"] = sharp_noaa_df["NOAA_AR"].astype(str).str.strip()

# --- Step 3: Merge SHARP/NOAA with DONKI (Flare, CME, SEP Data) ---
merged_df = sharp_noaa_df.merge(
    flare_df[["activeRegionNum", "flrID", "flareClass"]],
    left_on=["NOAA_AR"],
    right_on=["activeRegionNum"],
    how="left"
).drop(columns="activeRegionNum")

merged_df = merged_df.merge(
    cme_df[["activeRegionNum", "flrIDCme", "linkedCMETime"]],
    left_on=["NOAA_AR", "flrID"],
    right_on=["activeRegionNum", "flrIDCme"],
    how="left"
).drop(columns=["activeRegionNum", "flrIDCme"])

merged_df = merged_df.merge(
    sep_df[["activeRegionNum", "flrIDSep", "linkedSEPTime"]],
    left_on=["NOAA_AR", "flrID"],
    right_on=["activeRegionNum", "flrIDSep"],
    how="left"
).drop(columns=["activeRegionNum", "flrIDSep"])

# Convert columns to datetime format
merged_df["flrID"] = pd.to_datetime(merged_df["flrID"])
merged_df["T_REC"] = pd.to_datetime(merged_df["T_REC"])

# Apply the filter: Keep rows where either linkedSEPTime or linkedCMETime is within 12 hours after T_REC
filtered_df = merged_df[
    (
        ((merged_df["flrID"] - merged_df["T_REC"]).between(pd.Timedelta(0), pd.Timedelta(hours=12))) |
        (merged_df["flrID"].isna())
    ) |
    (
        merged_df["linkedSEPTime"].isna() & merged_df["linkedCMETime"].isna()
    )
]

filtered_df = filtered_df.iloc[:, 1:]

print(filtered_df.columns)

print("count:", filtered_df.shape[0])

filtered_df.to_csv("processed-data/SHARP_CME_SEP.csv", index=False)
print("Saved to processed-data/SHARP_CME_SEP.csv")