import requests
import json
import csv
import shutil
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode, explode_outer, size, split, to_timestamp, concat_ws, date_format, substr, expr
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

# Base URL for DONKI API
base_url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/"


# Function to fetch data from the API
def fetch_data(endpoint, params):
    url = f"{base_url}{endpoint}"
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None


# Function to save data to CSV
def save_to_csv(data, filename, fieldnames):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# Parameters for the date range from 2010 to 2019
params = {
    'startDate': '2010-01-01',
    'endDate': '2024-12-31'
}

# Fetch Solar Flare data
params['catalog'] = 'M2M_CATALOG'  # Default catalog
print('Downloading DONKI data...' )
flare_data = fetch_data("FLR", params)

# Define fieldnames for CSV files
flare_fieldnames = ['flrID', 'catalog', 'instruments', 'beginTime', 'peakTime', 'endTime', 'classType',
                    'sourceLocation', 'activeRegionNum', 'note', 'submissionTime', 'versionId', 'link', 'linkedEvents']

# Save Solar Flare data to CSV if available
if flare_data:
    flare_list = []
    for flare in flare_data:
        flare_entry = {key: flare.get(key, '') for key in flare_fieldnames}
        flare_list.append(flare_entry)

    save_to_csv(flare_list, 'raw-data/raw_DONKI_flare_CME_SEP_data_2010_2024.csv', flare_fieldnames)
    print("DONKI data saved to 'raw-data/raw_DONKI_flare_CME_SEP_data_2010_2024.csv'")

# Initialize Spark session
spark = SparkSession.builder.appName("FilterEvents").master("local[*]").getOrCreate()

# Define the schema of the array
linked_events_schema = ArrayType(StructType([StructField("activityID", StringType(), True)]))

print('Prepping DONKI data...')

# Load the dataset
df = spark.read.option("header", True).csv("raw-data/flare_CME_SEP_data_2010_2024.csv")
# print(df.show(truncate=False))

# Print initial row count before filtering
# initial_count = df.count()
# print(f"Initial count: {initial_count}")

# Filter out rows where 'linkedEvents' is null or empty (empty list)
#df_filtered = df.filter(col("linkedEvents").isNotNull() & (size(from_json(col("linkedEvents"), linked_events_schema)) > 0))

# Extract the first character from classType and store it in a new column flareClass
df = df.withColumn("flareClass", expr("substring(classType, 1, 1)"))

# Convert the 'linkedEvents' string to an array of JSON objects
df = df.withColumn("linkedEvents", from_json(col("linkedEvents"), linked_events_schema))

# Explode the array to get individual activityIDs
df_exploded = df.withColumn("linkedEvent", explode_outer(col("linkedEvents"))).drop("linkedEvents")

# Extract activityID from the linkedEvent struct
df_exploded = df_exploded.withColumn("activityID", col("linkedEvent.activityID"))

# Split activityID into linkedEventTime (timestamp) and linkedEventType (event type)
# Ensure proper timestamp formatting by concatenating the necessary components
df_exploded = df_exploded.withColumn(
    "linkedEventTime",
        concat_ws(
            "-",
            split(col("activityID"), "-")[0],  # Year
            split(col("activityID"), "-")[1],  # Month
            split(col("activityID"), "-")[2],   # Day
        )
)

# Convert to timestamp
df_exploded = df_exploded.withColumn(
    "linkedEventTime",
    date_format(to_timestamp(col("linkedEventTime"), "yyyy-MM-dd'T'HH:mm:ss"), "yyyy-MM-dd HH:mm:ss")
)
# Extract linkedEventType (event type) from the last part of activityID
df_exploded = df_exploded.withColumn("linkedEventType", split(col("activityID"), "-")[3]).drop("linkedEvent")

# Ensure proper timestamp formatting by concatenating the necessary components
df_exploded = df_exploded.withColumn(
    "flrID",
        concat_ws(
            "-",
            split(col("flrID"), "-")[0],  # Year
            split(col("flrID"), "-")[1],  # Month
            split(col("flrID"), "-")[2],   # Day
        )
)
# Convert to timestamp
df_exploded = df_exploded.withColumn(
    "flrID",
    date_format(to_timestamp(col("flrID"), "yyyy-MM-dd'T'HH:mm:ss"), "yyyy-MM-dd HH:mm:ss")
)

# Print row count after exploding
# exploded_count = df_exploded.count()
# print(f"Count after flattening and exploding (rows expanded): {exploded_count}")

# Show the flattened DataFrame (optional for verification)
df_exploded.show(truncate=False)

# Save the flattened result to a CSV file
df_exploded.write.option("header", "true").mode("overwrite").csv("expanded_DONKI_data_with_flare_CME_SEP.csv")

# Rename and move the file to the results directory
for file in os.listdir("flare_data_with_CME_SEP"):
    if file.endswith(".csv"):
        # Move the file to the results directory and rename it
        shutil.move(os.path.join("flare_data_with_CME_SEP", file), os.path.join("raw-data", "expanded_DONKI_data_with_flare_CME_SEP.csv"))
        break
# Delete the directory after moving the file
shutil.rmtree("flare_data_with_CME_SEP")

print('Data prep complete...')