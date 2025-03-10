import pandas as pd
import drms
from datetime import datetime

# Create a DRMS client instance
c = drms.Client()

# Query for data
query_hmi = 'hmi.sharp_cea_720s[]'
keys_hmi = 'T_REC, HARPNUM, NOAA_AR, MEANPOT, SHRGT45, TOTPOT, USFLUX, MEANJZH, ABSNJZH, SAVNCPP, MEANALP, MEANSHR, TOTUSJZ, TOTUSJH, MEANGAM, MEANGBZ, MEANJZD, AREA_ACR, R_VALUE, MEANGBT, MEANGBH, MEANGAM, CRLN_OBS, OBS_VR'
query_cgem = 'cgem.lorentz[]'
keys_cgem = 'T_REC, HARPNUM, NOAA_AR, TOTBSQ, TOTFZ, EPSZ'
conditions = '? (CRLN_OBS > -70) AND (CRLN_OBS < 70) AND (QUALITY = 0) AND (OBS_VR > -3500) AND (OBS_VR < 3500) ?'

# Set date ranges for batching (1 year at a time)
start_date = datetime(2018, 9, 1)
end_date = datetime(2024, 12, 31)

# Output CSV file
output_file = 'raw-data/combined_hmi_cgem_data.csv'

# Create an empty DataFrame to hold batch data
all_data = pd.DataFrame()
batch_count = 0  # Track number of batches processed

# Function to download data in batches
def download_data_in_batches(start_date, end_date):
    global all_data, batch_count
    current_start = start_date
    current_end = pd.to_datetime(current_start) + pd.DateOffset(months=2) - pd.DateOffset(days=1)

    while current_end <= end_date:
        print(f"Downloading data from {current_start.date()} to {current_end.date()}...")

        # Query for the current batch
        query_batch_hmi = f'{query_hmi}[{current_start.date()} - {current_end.date()}][{conditions}]'
        query_batch_cgem = f'{query_cgem}[{current_start.date()} - {current_end.date()}][{conditions}]'

        try:
            batch_data_hmi = c.query(query_batch_hmi, key=keys_hmi)
            batch_data_cgem = c.query(query_batch_cgem, key=keys_cgem)
        except Exception as e:
            print(f"Query failed for {current_start.date()} - {current_end.date()}: {e}")
            current_start = current_end + pd.DateOffset(days=1)
            current_end = pd.to_datetime(current_start) + pd.DateOffset(months=2) - pd.DateOffset(days=1)
            continue

        # Convert the batch data to a pandas DataFrame
        hmi_batch_df = pd.DataFrame(batch_data_hmi)
        cgem_batch_df = pd.DataFrame(batch_data_cgem)

        # Skip empty DataFrames
        if hmi_batch_df.empty and cgem_batch_df.empty:
            print(f"No data found for {current_start.date()} - {current_end.date()}, skipping...")
            current_start = current_end + pd.DateOffset(days=1)
            current_end = pd.to_datetime(current_start) + pd.DateOffset(months=2) - pd.DateOffset(days=1)
            continue

        # Merge data if both have data, otherwise concatenate
        if not hmi_batch_df.empty and not cgem_batch_df.empty:
            merged_batch_df = pd.merge(hmi_batch_df, cgem_batch_df, on=['T_REC', 'HARPNUM', 'NOAA_AR'], how='left')
        else:
            merged_batch_df = pd.concat([hmi_batch_df, cgem_batch_df], ignore_index=True)

        # Append the batch to the overall data
        all_data = pd.concat([all_data, merged_batch_df], ignore_index=True)
        batch_count += 1

        # Save progress after every 4 batches
        if batch_count % 4 == 0:
            write_to_csv()

        # Offload batch data from memory
        del hmi_batch_df
        del cgem_batch_df
        del merged_batch_df

        # Update the dates for the next batch
        current_start = current_end + pd.DateOffset(days=1)
        current_end = pd.to_datetime(current_start) + pd.DateOffset(months=2) - pd.DateOffset(days=1)

        # Final save if data remains
        if not all_data.empty:
            write_to_csv()

    print("Data download complete.")

# Function to write to CSV (appends after first write)
def write_to_csv():
    global all_data
    write_mode = 'w' if batch_count == 4 else 'a'  # First write: overwrite, subsequent writes: append
    header = batch_count == 4  # Write header only for the first batch

    all_data.to_csv(output_file, mode=write_mode, header=header, index=False)
    print(f"Data written to '{output_file}'. Total records saved: {len(all_data)}.")

    # Clear memory
    all_data = pd.DataFrame()

# Start the data download process
download_data_in_batches(start_date, end_date)