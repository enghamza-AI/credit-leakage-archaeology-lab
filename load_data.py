# load_data.py
 
import pandas as pd  
import os           
 
 
def load_data(filepath: str) -> pd.DataFrame:
 
 
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] File not found at: {filepath}"
            f"\nMake sure application_train.csv is inside your data/ folder."
        )
 
    print(f"[INFO] Loading data from: {filepath}")
 
    
    df = pd.read_csv(filepath)
 
    print(f"[INFO] Data loaded successfully.")
 
    
    run_sanity_check(df)
 
    return df  
 
 
def run_sanity_check(df: pd.DataFrame) -> None:
   
 
    print("\n" + "="*60)
    print("  DATASET SANITY CHECK")
    print("="*60)
 
    
    rows, cols = df.shape
    print(f"\n[SHAPE]")
    print(f"  Rows    : {rows:,}")   
    print(f"  Columns : {cols}")
 
    
    if 'TARGET' in df.columns:
        target_counts = df['TARGET'].value_counts()
        default_rate = df['TARGET'].mean() * 100  
 
        print(f"\n[TARGET COLUMN — What we are predicting]")
        print(f"  Repaid   (0) : {target_counts[0]:,}")
        print(f"  Defaulted(1) : {target_counts[1]:,}")
        print(f"  Default rate : {default_rate:.2f}%")
 
        
    else:
        print("\n[WARNING] No TARGET column found!")
 
    
    missing = df.isnull().sum()           
    missing = missing[missing > 0]        
    missing_pct = (missing / rows * 100).round(2)  
 
    print(f"\n[MISSING VALUES]")
    print(f"  Columns with missing data : {len(missing)} out of {cols}")
 
    if len(missing) > 0:
        
        worst = missing_pct.sort_values(ascending=False).head(5)
        print(f"  Top 5 worst columns:")
        for col, pct in worst.items():
            print(f"    {col:<40} {pct}% missing")
 
 
    dtype_counts = df.dtypes.value_counts()
 
    print(f"\n[COLUMN TYPES]")
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):<15} : {count} columns")
 
    
    duplicates = df.duplicated().sum()
    print(f"\n[DUPLICATES]")
    print(f"  Duplicate rows : {duplicates:,}")
    if duplicates > 0:
        print(f"  [WARNING] Duplicates found — potential Sin 5 leakage risk!")
 
    
    if 'SK_ID_CURR' in df.columns:
        unique_ids = df['SK_ID_CURR'].nunique()
        print(f"\n[BORROWER IDs]")
        print(f"  Total rows       : {rows:,}")
        print(f"  Unique borrower IDs : {unique_ids:,}")
        if unique_ids < rows:
            print(f"  [WARNING] Some borrowers appear more than once — group leakage risk!")
 
    print("\n" + "="*60)
    print("  SANITY CHECK COMPLETE")
    print("="*60 + "\n")
 
 

if __name__ == "__main__":
    
    data_path = os.path.join("data", "application_train.csv")
 
    
    df = load_data(data_path)
 

    print("[PREVIEW] First 5 rows:")
    print(df.head())
 