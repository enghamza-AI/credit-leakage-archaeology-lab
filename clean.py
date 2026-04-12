# clean.py

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data


def clean_data(df: pd.DataFrame,
               missing_threshold: float = 0.45,
               verbose: bool = True) -> pd.DataFrame:


    if verbose:
        print("\n" + "="*60)
        print("  DATA CLEANING")
        print("="*60)
        print(f"  Input shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    
    df_clean = df.copy()

    
    df_clean = _drop_high_missing_columns(df_clean, missing_threshold, verbose)

    
    df_clean = _handle_duplicates(df_clean, verbose)

    
    df_clean = _fix_known_quirks(df_clean, verbose)

    
    df_clean = _select_numeric_features(df_clean, verbose)

    
    if verbose:
        print(f"\n  Output shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
        print(f"  Columns removed: {df.shape[1] - df_clean.shape[1]}")
        print("\n  [REMINDER] Scaling and imputation go in the Pipeline AFTER splitting.")
        print("  If you do them here, you create leakage Sin 3.\n")
        print("="*60)
        print("  CLEANING COMPLETE")
        print("="*60)

    return df_clean


def _drop_high_missing_columns(df: pd.DataFrame,
                                threshold: float,
                                verbose: bool) -> pd.DataFrame:
   

    if verbose:
        print(f"\n── STEP 1: Drop high-missing columns (threshold: {threshold*100:.0f}%) ──")

    
    missing_frac = df.isnull().mean()  

    
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()

    
    cols_to_drop = [c for c in cols_to_drop if c not in ['TARGET', 'SK_ID_CURR']]

    if verbose:
        print(f"  Dropping {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing values:")
        for col in cols_to_drop[:10]:  # show first 10
            pct = missing_frac[col] * 100
            print(f"    {col:<45} {pct:.1f}% missing")
        if len(cols_to_drop) > 10:
            print(f"    ... and {len(cols_to_drop)-10} more")

    df = df.drop(columns=cols_to_drop)
    return df


def _handle_duplicates(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
   

    if verbose:
        print(f"\n── STEP 2: Handle duplicates ────────────────────────────")

    n_before = len(df)

    
    duplicates_mask = df.duplicated(keep='first')
    n_duplicates = duplicates_mask.sum()

    if n_duplicates > 0:
        df = df[~duplicates_mask]  
        if verbose:
            print(f"  Removed {n_duplicates:,} duplicate rows")
            print(f"  {n_before:,} → {len(df):,} rows remaining")
    else:
        if verbose:
            print(f"  No duplicate rows found — dataset is clean on this dimension")

    return df


def _fix_known_quirks(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
  

    if verbose:
        print(f"\n── STEP 3: Fix known data quirks ────────────")

    
    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs()  
        df['AGE_YEARS'] = (df['DAYS_BIRTH'] / 365.25).round(1)
        if verbose:
            print(f"  DAYS_BIRTH: converted to positive, added AGE_YEARS column")

    
    if 'DAYS_EMPLOYED' in df.columns:
        anomaly_count = (df['DAYS_EMPLOYED'] == 365243).sum()
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs()  
        if verbose:
            print(f"  DAYS_EMPLOYED: replaced {anomaly_count:,} anomalous values (365243 → NaN)")
            print(f"  DAYS_EMPLOYED: converted to positive days")

    return df


def _select_numeric_features(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
  

    if verbose:
        print(f"\n── STEP 4: Select numeric features ─────────────────────")

    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if verbose:
        print(f"  Keeping  : {len(numeric_cols)} numeric columns (includes TARGET, SK_ID_CURR)")
        print(f"  Dropping : {len(categorical_cols)} categorical columns")
        if categorical_cols:
            print(f"  Dropped categoricals: {categorical_cols[:5]}{'...' if len(categorical_cols)>5 else ''}")
        print(f"  [NOTE] Categoricals dropped here for safety.")
        print(f"         Sin 1 demo will show what happens when you encode them wrong.")

    
    df = df[numeric_cols]

    return df


def get_features_and_target(df_clean: pd.DataFrame):
  

    
    exclude_cols = ['TARGET', 'SK_ID_CURR']
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]

    X = df_clean[feature_cols] 
    y = df_clean['TARGET']      

    return X, y



if __name__ == "__main__":
    data_path = os.path.join("data", "application_train.csv")
    df = load_data(data_path)
    df_clean = clean_data(df, verbose=True)

    X, y = get_features_and_target(df_clean)

    print(f"\n[RESULT]")
    print(f"  Features (X) shape : {X.shape}")
    print(f"  Target   (y) shape : {y.shape}")
    print(f"  Default rate       : {y.mean()*100:.2f}%")
    print(f"\nFirst 5 feature names:")
    print(X.columns.tolist()[:5])