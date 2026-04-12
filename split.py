

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,          
    StratifiedKFold,           
    GroupKFold,                
)
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data
from clean import clean_data, get_features_and_target


def random_split(X: pd.DataFrame,
                 y: pd.Series,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 verbose: bool = True):


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        
        stratify=y
    )

    if verbose:
        _print_split_stats("RANDOM SPLIT", X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def stratified_split(X: pd.DataFrame,
                     y: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     verbose: bool = True):
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   
    )

    if verbose:
        _print_split_stats("STRATIFIED SPLIT", X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def group_kfold_split(X: pd.DataFrame,
                      y: pd.Series,
                      groups: pd.Series,
                      n_splits: int = 5,
                      verbose: bool = True):
 

    if verbose:
        print(f"\n── GROUP K-FOLD SPLIT ───────────────────────────────────")
        print(f"  n_splits   : {n_splits}")
        print(f"  Total rows : {len(X):,}")
        print(f"  Unique groups (borrowers): {groups.nunique():,}")
        print(f"  Groups per fold will NOT overlap between train and test")

    gkf = GroupKFold(n_splits=n_splits)

    splits = []

    for fold_num, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        
        X_train_fold = X.iloc[train_idx]
        X_test_fold  = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold  = y.iloc[test_idx]

        if verbose and fold_num == 0:
          
            _print_split_stats(f"FOLD 1 of {n_splits}", X_train_fold, X_test_fold,
                               y_train_fold, y_test_fold)

            
            train_groups = set(groups.iloc[train_idx])
            test_groups  = set(groups.iloc[test_idx])
            overlap = train_groups.intersection(test_groups)
            print(f"\n  [VERIFICATION] Borrower overlap between train/test: {len(overlap)}")
            if len(overlap) == 0:
                print(f"   ZERO overlap — no borrower appears in both train and test")
            else:
                print(f"   OVERLAP FOUND — {len(overlap)} borrowers in both sets!")

        splits.append((X_train_fold, X_test_fold, y_train_fold, y_test_fold))

    if verbose:
        print(f"\n  All {n_splits} folds created.")
        print(f"  Use the average AUC across folds as your final score.")

    return splits


def demonstrate_leaky_vs_honest_split(X: pd.DataFrame,
                                      y: pd.Series,
                                      groups: pd.Series,
                                      verbose: bool = True):
    

    if verbose:
        print("\n" + "="*60)
        print("  SPLIT COMPARISON: LEAKY vs HONEST")
        print("="*60)
        print("\n[LEAKY] Using random split — borrowers can appear in both sets")

    leaky_split = random_split(X, y, verbose=verbose)

    if verbose:
        print("\n[HONEST] Using Group K-Fold — borrowers stay in one set only")

    honest_splits = group_kfold_split(X, y, groups, verbose=verbose)

    return leaky_split, honest_splits


def _print_split_stats(label: str,
                       X_train, X_test,
                       y_train, y_test) -> None:
    """Helper: prints train/test split statistics."""

    total = len(X_train) + len(X_test)

    print(f"\n── {label} ──")
    print(f"  Train : {len(X_train):>7,} rows ({len(X_train)/total*100:.1f}%)")
    print(f"  Test  : {len(X_test):>7,} rows ({len(X_test)/total*100:.1f}%)")
    print(f"  Train default rate : {y_train.mean()*100:.2f}%")
    print(f"  Test  default rate : {y_test.mean()*100:.2f}%")

    
    diff = abs(y_train.mean() - y_test.mean()) * 100
    if diff < 0.5:
        print(f"   Class balance maintained (diff: {diff:.3f}%)")
    else:
        print(f"    Class balance differs by {diff:.3f}%")



if __name__ == "__main__":
    data_path = os.path.join("data", "application_train.csv")

    print("[INFO] Loading and cleaning data...")
    df = load_data(data_path)
    df_clean = clean_data(df, verbose=False)
    X, y = get_features_and_target(df_clean)

    print(f"\n[INFO] Features: {X.shape[1]} columns, {X.shape[0]:,} rows")

    
    X_train, X_test, y_train, y_test = random_split(X, y, verbose=True)

    
    X_train, X_test, y_train, y_test = stratified_split(X, y, verbose=True)

    
    groups = df_clean['SK_ID_CURR'] if 'SK_ID_CURR' in df_clean.columns else pd.Series(range(len(X)))
    folds = group_kfold_split(X, y, groups, n_splits=5, verbose=True)

    print(f"\n[DONE] Split demonstrations complete.")
    print(f"  Next step: train.py — train a model on these splits and measure AUC")