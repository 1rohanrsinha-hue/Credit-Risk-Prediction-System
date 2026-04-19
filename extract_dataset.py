# extract_dataset.py - Extract Lending Club dataset
import gzip
import shutil
import os

print("="*60)
print("EXTRACTING LENDING CLUB DATASET")
print("="*60)

# Source file
source_file = "accepted_2007_to_2018Q4.csv.gz"

# Check if file exists
if os.path.exists(source_file):
    print(f"✅ Found file: {source_file}")
    
    # Create data folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("📁 Created 'data' folder")
    
    # Destination file
    dest_file = "data/lending_club_loan.csv"
    
    print(f"📦 Extracting {source_file}...")
    
    # Extract the gz file
    with gzip.open(source_file, 'rb') as f_in:
        with open(dest_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"✅ Dataset extracted to: {dest_file}")
    
    # Check file size
    file_size = os.path.getsize(dest_file) / (1024 * 1024)  # Convert to MB
    print(f"📊 File size: {file_size:.2f} MB")
    
    # Read first few rows to verify
    import pandas as pd
    print("\n📋 Preview of first 5 rows:")
    df_sample = pd.read_csv(dest_file, nrows=5)
    print(df_sample[['loan_amnt', 'int_rate', 'annual_inc', 'loan_status']].head())
    
    print("\n✅ Dataset is ready to use!")
else:
    print(f"❌ File not found: {source_file}")
    print("Please make sure the file is in the same folder as this script")