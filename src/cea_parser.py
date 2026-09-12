import pandas as pd
import json
import os

def parse_cea_capacity_limits(excel_file_path):
    """
    Parses the messy CEA Daily Report Excel file to extract the 
    official Installed Capacity (MW) limits for Renewable Energy (R.E.S).
    """
    print(f"Loading official CEA Grid Data from: {excel_file_path}...")
    
    try:
        # Load the Excel file. We skip the first 3 rows of massive title text 
        # but keep the actual table headers to maintain column alignment.
        df = pd.read_excel(excel_file_path, skiprows=3)
        
        # 1. Scan the first column (index 0) for the R.E.S label
        # We convert to string, remove dots/spaces, and make uppercase to catch any weird formatting
        clean_first_col = df.iloc[:, 0].astype(str).str.replace(r'[\.\s]', '', regex=True).str.upper()
        
        # Find the row index where 'RES' exists
        res_row_indices = df[clean_first_col.str.contains("RES|RENEWABLE", na=False)].index
        
        if len(res_row_indices) == 0:
             print("Warning: Could not automatically detect R.E.S row. Format mismatch!")
             return None
             
        target_idx = res_row_indices[0]
        
        # 2. Extract the capacity from the 'Installed Capacity (MW)' column
        # Because headers might be merged, the safest bet is grabbing the value
        # from the 2nd column (index 1) of that specific row.
        raw_capacity = str(df.iloc[target_idx, 1])
        
        # 3. Clean the number (remove commas like '239,660.98') and convert to float
        official_mw_limit = float(raw_capacity.replace(',', '').strip())
        
        print(f"SUCCESS: Extracted Official Indian R.E.S Capacity Limit: {official_mw_limit:,.2f} MW")
        return official_mw_limit

    except Exception as e:
        print(f"Error parsing CEA data: {e}")
        return None

def build_sample_sites_json(official_capacity_mw):
    """
    Builds the sample_sites.json file needed for the Streamlit decision matrix,
    grounding the simulated solar plant alerts in real-world CEA capacity scales.
    """
    # Create realistic sub-station limits based on the massive national capacity
    sites_config = {
        "Western_Region_Solar_Park": {
            "max_capacity_mw": official_capacity_mw * 0.05, # Simulating 5% of national capacity
            "curtailment_warning_threshold": 0.85 # Alert when generation hits 85% of limit
        },
        "Southern_Region_Wind_Farm": {
            "max_capacity_mw": official_capacity_mw * 0.03,
            "curtailment_warning_threshold": 0.90
        }
    }
    
    # Save the configuration
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sample_sites.json')
    with open(output_path, 'w') as f:
        json.dump(sites_config, f, indent=4)
        
    print(f"Generated realistic grid constraints at: {output_path}")

# === Execution ===
if __name__ == "__main__":
    # Point to the downloaded file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cea_file = os.path.join(BASE_DIR, 'data', 'cea_daily_report.xls')
    
    extracted_limit = parse_cea_capacity_limits(cea_file)
    
    if extracted_limit:
        build_sample_sites_json(extracted_limit)