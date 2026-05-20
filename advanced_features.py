# ============================================================================
# advanced_features.py — PGP-Level Feature Engineering for Sensor Data
# ============================================================================
"""
Advanced feature engineering module demonstrating signal processing and
domain-specific knowledge for the Predictive Maintenance capstone.

Features computed:
1. Time-Domain Statistics: RMS, Kurtosis, Skewness, Crest Factor
2. Spectral Decomposition: DFT magnitude spectrum of sensor profile
3. Cross-Sensor Interactions: Pairwise ratios, differences, products
4. Domain-Specific: Thermal efficiency, pressure gradients, anomaly z-scores
5. Legacy Features: Temp_Pressure_Ratio, Coolant_Efficiency, High_RPM_Flag
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft


def compute_time_domain_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Compute time-domain statistical features across the sensor vector.

    Treats each sample's sensor readings as a discrete signal and computes
    statistical moments that capture the 'shape' of the sensor profile.

    Args:
        df: DataFrame with sensor columns
        sensor_cols: List of sensor column names

    Returns:
        DataFrame with new time-domain features
    """
    sensor_matrix = df[sensor_cols].values

    # Root Mean Square — captures overall signal amplitude
    # RMS = sqrt(mean(x²))
    rms = np.sqrt(np.mean(sensor_matrix ** 2, axis=1))

    # Kurtosis — measures 'peakedness' of sensor distribution
    # High kurtosis → concentrated extreme values → potential degradation
    kurt = stats.kurtosis(sensor_matrix, axis=1, fisher=True)

    # Skewness — measures asymmetry in sensor distribution
    # Non-zero skew → sensor readings pulling in one direction → imbalance
    skew = stats.skew(sensor_matrix, axis=1)

    # Peak-to-Peak — range of sensor values per sample
    # Widening range indicates wear/instability
    peak_to_peak = np.ptp(sensor_matrix, axis=1)

    # Crest Factor — ratio of peak to RMS
    # High crest factor → sharp spikes in sensor readings → anomaly indicator
    peak_abs = np.max(np.abs(sensor_matrix), axis=1)
    crest_factor = np.divide(peak_abs, rms, out=np.zeros_like(rms), where=rms != 0)

    # Standard deviation of sensor readings
    sensor_std = np.std(sensor_matrix, axis=1)

    # Coefficient of Variation — normalized dispersion
    sensor_mean = np.mean(sensor_matrix, axis=1)
    cv = np.divide(sensor_std, np.abs(sensor_mean),
                   out=np.zeros_like(sensor_mean), where=sensor_mean != 0)

    result = pd.DataFrame({
        'Sensor_RMS': rms,
        'Sensor_Kurtosis': kurt,
        'Sensor_Skewness': skew,
        'Sensor_PeakToPeak': peak_to_peak,
        'Sensor_CrestFactor': crest_factor,
        'Sensor_Std': sensor_std,
        'Sensor_CV': cv,
    }, index=df.index)

    return result


def compute_spectral_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Compute frequency-domain features via DFT of the sensor profile.

    Each sample's 6 sensor readings are treated as a discrete signal.
    The DFT decomposes this into frequency components, revealing the
    'spectral signature' of the engine state.

    For a 6-point signal, the DFT yields 3 unique magnitude components
    (due to conjugate symmetry), plus the DC component (mean).

    Args:
        df: DataFrame with sensor columns
        sensor_cols: List of sensor column names

    Returns:
        DataFrame with spectral features
    """
    sensor_matrix = df[sensor_cols].values
    n_samples = sensor_matrix.shape[0]
    n_sensors = len(sensor_cols)

    # Normalize each sample to zero-mean before FFT
    centered = sensor_matrix - sensor_matrix.mean(axis=1, keepdims=True)

    # Compute FFT along sensor axis
    fft_result = fft(centered, axis=1)
    magnitudes = np.abs(fft_result)

    # Extract unique frequency components (Nyquist: n//2 + 1)
    n_unique = n_sensors // 2 + 1  # = 4 for 6 sensors

    features = {}

    # DC component (should be ~0 after centering, but include for completeness)
    features['FFT_DC'] = magnitudes[:, 0]

    # Frequency bin magnitudes (bins 1 to n//2)
    for i in range(1, n_unique):
        features[f'FFT_Mag_{i}'] = magnitudes[:, i]

    # Total spectral energy (Parseval's theorem)
    features['Spectral_Energy'] = np.sum(magnitudes[:, 1:n_unique] ** 2, axis=1)

    # Spectral centroid — 'center of mass' of the spectrum
    freq_bins = np.arange(1, n_unique)
    mag_slice = magnitudes[:, 1:n_unique]
    total_mag = mag_slice.sum(axis=1, keepdims=True)
    total_mag = np.where(total_mag == 0, 1, total_mag)  # avoid division by zero
    features['Spectral_Centroid'] = (mag_slice * freq_bins).sum(axis=1) / total_mag.ravel()

    # Dominant frequency — which bin has the highest energy
    features['Dominant_Freq_Bin'] = np.argmax(mag_slice, axis=1) + 1

    return pd.DataFrame(features, index=df.index)


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific cross-sensor interaction features.

    These capture physical relationships between engine subsystems:
    - Temperature vs. Pressure ratios (thermal efficiency)
    - Pressure gradients (flow resistance)
    - RPM-normalized readings (per-revolution metrics)

    Args:
        df: DataFrame with renamed sensor columns

    Returns:
        DataFrame with interaction features
    """
    features = {}

    # ── Legacy Features (preserved for backward compatibility) ──
    features['Temp_Pressure_Ratio'] = df['Lub_Oil_Temperature'] / df['Lub_Oil_Pressure'].replace(0, np.nan)
    features['Coolant_Efficiency'] = df['Coolant_Pressure'] / df['Coolant_Temperature'].replace(0, np.nan)

    # ── Thermal Features ──
    # Temperature differential — indicator of heat dissipation efficiency
    features['Temp_Differential'] = df['Lub_Oil_Temperature'] - df['Coolant_Temperature']

    # Thermal load index — combined temperature stress
    features['Thermal_Load'] = (df['Lub_Oil_Temperature'] + df['Coolant_Temperature']) / 2

    # ── Pressure Features ──
    # Pressure gradient — differential across lubrication and fuel systems
    features['Pressure_Gradient'] = df['Lub_Oil_Pressure'] - df['Fuel_Pressure']

    # Total system pressure — overall hydraulic stress
    features['Total_Pressure'] = (df['Lub_Oil_Pressure'] + df['Fuel_Pressure'] +
                                   df['Coolant_Pressure'])

    # ── RPM-Normalized Metrics ──
    # Pressure per RPM — efficiency metrics
    rpm_safe = df['Engine_RPM'].replace(0, np.nan)
    features['Oil_Press_per_RPM'] = df['Lub_Oil_Pressure'] / rpm_safe * 1000

    # ── Z-Score Anomaly Indicators ──
    # How many std deviations each sensor is from its mean
    for col in ['Engine_RPM', 'Lub_Oil_Pressure', 'Lub_Oil_Temperature']:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std > 0:
            features[f'{col}_ZScore'] = (df[col] - col_mean) / col_std

    result = pd.DataFrame(features, index=df.index)

    # Fill NaN values with column medians
    for col in result.columns:
        median_val = result[col].median()
        result[col] = result[col].fillna(median_val if not np.isnan(median_val) else 0)

    return result


def compute_high_rpm_flag(df: pd.DataFrame, percentile: float = 0.85) -> pd.Series:
    """
    Compute High RPM flag — binary indicator for extreme RPM readings.

    Args:
        df: DataFrame with Engine_RPM column
        percentile: Threshold percentile (default: 85th)

    Returns:
        Series with binary High_RPM_Flag
    """
    threshold = df['Engine_RPM'].quantile(percentile)
    return (df['Engine_RPM'] > threshold).astype(int)


def engineer_all_features(df: pd.DataFrame,
                           sensor_cols: list = None,
                           high_rpm_percentile: float = 0.85) -> pd.DataFrame:
    """
    Master function: Compute all advanced features for the dataset.

    Orchestrates time-domain, spectral, interaction, and flag features
    into a single enriched DataFrame.

    Args:
        df: Raw DataFrame with sensor columns (already renamed)
        sensor_cols: List of raw sensor column names
        high_rpm_percentile: Percentile for High RPM flag

    Returns:
        DataFrame with all original + engineered features
    """
    if sensor_cols is None:
        sensor_cols = [
            'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
            'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature'
        ]

    # Verify columns exist
    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    # Normalize sensor values for time-domain and spectral analysis
    # (use StandardScaler-like normalization within compute functions)

    # 1. Time-domain features
    time_feats = compute_time_domain_features(df, sensor_cols)

    # 2. Spectral features
    spectral_feats = compute_spectral_features(df, sensor_cols)

    # 3. Interaction features (includes legacy features)
    interaction_feats = compute_interaction_features(df)

    # 4. High RPM flag
    high_rpm = compute_high_rpm_flag(df, high_rpm_percentile)

    # Combine all features
    result = pd.concat([
        df[sensor_cols],           # Original 6 sensors
        time_feats,                # 7 time-domain features
        spectral_feats,            # 8 spectral features
        interaction_feats,         # ~12 interaction features
        high_rpm.rename('High_RPM_Flag')  # 1 binary flag
    ], axis=1)

    # Final cleanup: replace any remaining inf/nan
    result = result.replace([np.inf, -np.inf], np.nan)
    for col in result.columns:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())

    print(f"[OK] Feature engineering complete: {result.shape[1]} features "
          f"({len(sensor_cols)} raw + {result.shape[1] - len(sensor_cols)} engineered)")

    return result


def get_feature_descriptions() -> dict:
    """Return human-readable descriptions of all engineered features."""
    return {
        # Time-domain
        'Sensor_RMS': 'Root Mean Square of sensor readings — overall signal amplitude',
        'Sensor_Kurtosis': 'Kurtosis of sensor distribution — peakedness indicator',
        'Sensor_Skewness': 'Skewness of sensor distribution — asymmetry measure',
        'Sensor_PeakToPeak': 'Peak-to-peak range — signal spread indicator',
        'Sensor_CrestFactor': 'Peak / RMS ratio — spike severity indicator',
        'Sensor_Std': 'Standard deviation across sensors — variability measure',
        'Sensor_CV': 'Coefficient of variation — normalized dispersion',
        # Spectral
        'FFT_DC': 'DC component of sensor DFT — mean level',
        'FFT_Mag_1': 'First harmonic magnitude — primary oscillation',
        'FFT_Mag_2': 'Second harmonic magnitude — secondary pattern',
        'FFT_Mag_3': 'Third harmonic magnitude — higher-order pattern',
        'Spectral_Energy': 'Total spectral energy — overall signal power',
        'Spectral_Centroid': 'Spectral center of mass — frequency balance',
        'Dominant_Freq_Bin': 'Dominant frequency bin — primary pattern index',
        # Interactions
        'Temp_Pressure_Ratio': 'Oil Temperature / Oil Pressure — thermal stress',
        'Coolant_Efficiency': 'Coolant Pressure / Coolant Temperature — cooling efficiency',
        'Temp_Differential': 'Oil Temp − Coolant Temp — heat dissipation',
        'Thermal_Load': 'Average temperature — overall thermal stress',
        'Pressure_Gradient': 'Oil Pressure − Fuel Pressure — flow resistance',
        'Total_Pressure': 'Sum of all pressures — hydraulic stress',
        'Oil_Press_per_RPM': 'Oil Pressure normalized by RPM — per-revolution efficiency',
        'Engine_RPM_ZScore': 'RPM z-score — statistical anomaly indicator',
        'Lub_Oil_Pressure_ZScore': 'Oil Pressure z-score — statistical anomaly',
        'Lub_Oil_Temperature_ZScore': 'Oil Temperature z-score — statistical anomaly',
        'High_RPM_Flag': 'Binary flag: RPM > 85th percentile',
    }


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with sample data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'Engine_RPM': np.random.uniform(300, 2000, 100),
        'Lub_Oil_Pressure': np.random.uniform(1, 7, 100),
        'Fuel_Pressure': np.random.uniform(1, 20, 100),
        'Coolant_Pressure': np.random.uniform(0.5, 5, 100),
        'Lub_Oil_Temperature': np.random.uniform(60, 100, 100),
        'Coolant_Temperature': np.random.uniform(60, 100, 100),
    })

    result = engineer_all_features(test_data)
    print(f"\nOutput shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")
    print(f"\nSample:\n{result.head(3).T}")
    print(f"\nNo NaN: {not result.isna().any().any()}")
    print(f"No Inf: {not np.isinf(result.values).any()}")
