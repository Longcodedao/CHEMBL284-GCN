from chembl_webresource_client.new_client import new_client
import pandas as pd
from tqdm import tqdm

# Initialize the ChEMBL client
chembl = new_client

# Search for target CHEMBL284
target = chembl.target
target_data = target.filter(target_chembl_id="CHEMBL284").only(['target_chembl_id', 'pref_name'])

# Print target information
for t in target_data:
    print("Target Information:", t)

# Fetch bioactivity data for CHEMBL284
activity = chembl.activity

# Initialize an empty list to store data
bioactivity_list = []

# Use tqdm for progress bar while fetching data
print("Fetching bioactivity data...")
for record in tqdm(activity.filter(target_chembl_id="CHEMBL284").only([
    "molecule_chembl_id", "assay_chembl_id", "standard_type", 
    "standard_value", "standard_units", "canonical_smiles"
])):
    bioactivity_list.append(record)

# Convert the list of records to a DataFrame
bioactivity_df = pd.DataFrame(bioactivity_list)

# Save the data to a CSV file
bioactivity_df.to_csv("CHEMBL284_bioactivity.csv", index=False)
print("Bioactivity data saved to CHEMBL284_bioactivity.csv")


