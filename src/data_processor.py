import pandas as pd
import numpy as np
import os

# ==========================================
# 1. DYNAMIC PATH SETUP
# ==========================================
# This dynamically routes to your 'data' folder from inside 'src'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# File paths
GEN_PATH = os.path.join(DATA_DIR, 'Plant_1_Generation_Data.csv')
SENSOR_PATH = os.path.join(DATA_DIR, 'Plant_1_Weather_Sensor_Data.csv')
NASA_PATH = os.path.join(DATA_DIR, 'nasa_power.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'synthetic_historical.csv')

def load_and_merge():
    """
    Loads the real-time weather and generation datasets and merges them 
    on their synchronized timestamps.
    """
    print("Loading Kaggle and NASA datasets...")
    
    # Load Kaggle datasets
    df_gen = pd.read_csv(GEN_PATH)
    df_sensor = pd.read_csv(SENSOR_PATH)
    
    # Standardize timestamp columns to datetime objects (FIX: Add dayfirst=True)
    df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], dayfirst=True, format='mixed')
    df_sensor['DATE_TIME'] = pd.to_datetime(df_sensor['DATE_TIME'], dayfirst=True, format='mixed')
    
    # Merge Kaggle data on Timestamp and Plant ID
    df_merged = pd.merge(df_gen, df_sensor, on=['DATE_TIME', 'PLANT_ID'], how='inner')
    df_merged.rename(columns={'DATE_TIME': 'Timestamp'}, inplace=True)
    
    # Load NASA data if available and merge
    if os.path.exists(NASA_PATH):
        df_nasa = pd.read_csv(NASA_PATH)
        # Ensure the NASA timestamp is also a datetime object
        if 'Timestamp' in df_nasa.columns:
            df_nasa['Timestamp'] = pd.to_datetime(df_nasa['Timestamp'])
            df_merged = pd.merge(df_merged, df_nasa, on='Timestamp', how='inner')
            print("Successfully joined NASA meteorological data.")
        
    return df_merged

def clean_and_engineer_features(df):
    """
    Handles missing values, caps outliers, and engineers temporal features.
    """
    print("Applying data engineering pipeline...")
    
    # Sort chronologically to ensure time-series integrity
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # A. Handle Missing Time-Steps (FIX: Only interpolate numerical columns)
    # This prevents the 'Cannot interpolate with str dtype' crash
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
    
    # B. Cap Anomalous Outliers (IQR Method)
    cols_to_cap = ['DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'PM2.5', 'AT', 'RH', 'WS', 'NO2']
    for col in cols_to_cap:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            
    # C. Feature Engineering (Temporal Data)
    df['Hour'] = df['Timestamp'].dt.hour
    df['Month'] = df['Timestamp'].dt.month
    
    return df

# === Execution ===
if __name__ == "__main__":
    try:
        # 1. Merge all sources
        raw_df = load_and_merge()
        
        # 2. Clean and engineer
        clean_df = clean_and_engineer_features(raw_df)
        
        # 3. Save the final dataset for the ML engine
        clean_df.to_csv(OUTPUT_PATH, index=False)
        print(f"SUCCESS: Pipeline complete. Usable dataset saved to {OUTPUT_PATH}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure your CSV files are placed in the 'data' folder.")