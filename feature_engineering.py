import pandas as pd
import numpy as np

def generate_rolling_features(input_csv_path='data/engine_data.csv', output_csv_path='data/engine_data_rolling.csv'):
    print("Initializing rolling window feature engineering...")
    df = pd.read_csv(input_csv_path)
    
    # Check current columns and rename if necessary
    col_map = {
        'Engine rpm': 'Engine_RPM',
        'Lub oil pressure': 'Lub_Oil_Pressure',
        'Fuel pressure': 'Fuel_Pressure',
        'Coolant pressure': 'Coolant_Pressure',
        'lub oil temp': 'Lub_Oil_Temperature',
        'Coolant temp': 'Coolant_Temperature',
        'Engine Condition': 'Engine_Condition'
    }
    df.rename(columns=col_map, inplace=True)
    
    # Create a synthetic timestamp assuming 1 reading per minute
    df['Timestamp'] = pd.date_range(start='2026-01-01', periods=len(df), freq='1T')
    df.set_index('Timestamp', inplace=True)
    
    # Calculate 5-minute rolling average for key sensors
    print("Calculating 5-minute rolling averages for Engine_RPM and Temperatures...")
    df['Engine_RPM_5m_avg'] = df['Engine_RPM'].rolling(window=5, min_periods=1).mean()
    df['Lub_Oil_Temp_5m_avg'] = df['Lub_Oil_Temperature'].rolling(window=5, min_periods=1).mean()
    
    # Calculate rolling standard deviations to capture volatility
    df['Engine_RPM_5m_std'] = df['Engine_RPM'].rolling(window=5, min_periods=1).std().fillna(0)
    
    # Reset index to bring Timestamp back as a column
    df.reset_index(inplace=True)
    
    # Save the feature-engineered dataset
    df.to_csv(output_csv_path, index=False)
    print(f"Feature engineering complete. Saved to {output_csv_path}.")

if __name__ == "__main__":
    generate_rolling_features()
