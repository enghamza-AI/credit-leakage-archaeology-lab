# eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import os


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data


def run_eda(df: pd.DataFrame, save_plots: bool = True) -> None:
   
    os.makedirs("outputs", exist_ok=True)

    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)

    
    _analyze_target(df, save_plots)

    
    _analyze_feature_types(df)

    
    _analyze_missing_values(df, save_plots)

    
    _hunt_suspicious_correlations(df, save_plots)

    
    _analyze_key_features(df, save_plots)

    print("\n[EDA COMPLETE] Charts saved to outputs/ folder\n")


def _analyze_target(df: pd.DataFrame, save_plots: bool) -> None:
    """Analyzes and plots the target variable distribution."""

    print("\n── TARGET DISTRIBUTION ──────────────────────────────────")

    counts = df['TARGET'].value_counts()
    total = len(df)
    default_pct = counts[1] / total * 100
    repaid_pct = counts[0] / total * 100

    print(f"  Repaid   (0): {counts[0]:>7,}  ({repaid_pct:.1f}%)")
    print(f"  Defaulted(1): {counts[1]:>7,}  ({default_pct:.1f}%)")
    print(f"\n  → Dataset is IMBALANCED: only {default_pct:.1f}% defaults")
    print(f"  → This is WHY we use AUC not accuracy.")
    print(f"    A model that always predicts 'repaid' would get {repaid_pct:.1f}% accuracy")
    print(f"    but AUC would be 0.50 — exposing it as useless.")

    if save_plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Repaid (0)', 'Defaulted (1)'], [counts[0], counts[1]],
               color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.5)
        ax.set_title('Target Distribution — Home Credit Dataset', fontweight='bold')
        ax.set_ylabel('Number of Applications')

        
        for i, (label, val) in enumerate(zip(['Repaid', 'Defaulted'], [counts[0], counts[1]])):
            pct = val / total * 100
            ax.text(i, val + 1000, f'{pct:.1f}%', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('outputs/01_target_distribution.png', dpi=150)
        plt.close()
        print(f"  [SAVED] outputs/01_target_distribution.png")


def _analyze_feature_types(df: pd.DataFrame) -> None:
    """Breaks down columns by their type and role."""

    print("\n── FEATURE TYPES ────────────────────────────────────────")

    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    
    feature_numeric = [c for c in numeric_cols if c not in ['TARGET', 'SK_ID_CURR']]
    id_cols = ['SK_ID_CURR'] if 'SK_ID_CURR' in df.columns else []

    print(f"  ID columns       : {len(id_cols)}  → {id_cols}")
    print(f"  Target column    : 1   → ['TARGET']")
    print(f"  Numeric features : {len(feature_numeric)}")
    print(f"  Category features: {len(categorical_cols)}")
    print(f"  TOTAL columns    : {df.shape[1]}")

    if categorical_cols:
        print(f"\n  Categorical columns (will need encoding later):")
        for col in categorical_cols[:10]:  
            n_unique = df[col].nunique()
            print(f"    {col:<40} {n_unique} unique values")
        if len(categorical_cols) > 10:
            print(f"    ... and {len(categorical_cols)-10} more")


def _analyze_missing_values(df: pd.DataFrame, save_plots: bool) -> None:
    """Finds and visualizes missing value patterns."""

    print("\n── MISSING VALUES ───────────────────────────────────────")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(1)

    print(f"  {len(missing)} columns have missing data out of {df.shape[1]} total")

    
    severe   = missing_pct[missing_pct > 40]   
    moderate = missing_pct[(missing_pct > 10) & (missing_pct <= 40)]
    mild     = missing_pct[missing_pct <= 10]

    print(f"\n  Severe   (>40% missing) : {len(severe)} columns")
    print(f"  Moderate (10-40% missing): {len(moderate)} columns")
    print(f"  Mild     (<10% missing)  : {len(mild)} columns")

    print(f"\n  Top 10 most missing columns:")
    for col, pct in missing_pct.head(10).items():
        bar = "█" * int(pct / 5)  
        print(f"    {col:<40} {pct:5.1f}% {bar}")

    if save_plots and len(missing_pct) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        top30 = missing_pct.head(30)
        colors = ['#e74c3c' if p > 40 else '#f39c12' if p > 10 else '#3498db'
                  for p in top30.values]
        ax.barh(top30.index, top30.values, color=colors)
        ax.set_xlabel('% Missing')
        ax.set_title('Top 30 Columns by Missing Data %', fontweight='bold')
        ax.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='40% threshold')
        ax.legend()
        plt.tight_layout()
        plt.savefig('outputs/02_missing_values.png', dpi=150)
        plt.close()
        print(f"\n  [SAVED] outputs/02_missing_values.png")


def _hunt_suspicious_correlations(df: pd.DataFrame, save_plots: bool) -> None:
 

    print("\n── LEAKAGE HUNT: SUSPICIOUS CORRELATIONS ────────────────")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    
    feature_cols = [c for c in numeric_cols if c not in ['TARGET', 'SK_ID_CURR']]

    
    correlations = df[feature_cols].corrwith(df['TARGET']).abs()  # abs = ignore sign
    correlations = correlations.sort_values(ascending=False)

    print(f"  Checking {len(feature_cols)} numeric features for suspicious correlation with TARGET...")

    
    suspicious = correlations[correlations > 0.5]
    high = correlations[correlations > 0.3]

    print(f"\n  Top 15 features by correlation with TARGET:")
    for col, corr in correlations.head(15).items():
        flag = " ← [SUSPICIOUS - possible Sin 2!]" if corr > 0.5 else ""
        warn = " ← [HIGH - worth investigating]" if 0.3 < corr <= 0.5 else ""
        print(f"    {col:<40} {corr:.4f}{flag}{warn}")

    if len(suspicious) > 0:
        print(f"\n  [WARNING] {len(suspicious)} features have correlation > 0.5 with TARGET")
        print(f"  These are candidates for Sin 2 (Feature From Target) investigation")
    else:
        print(f"\n  [CLEAN] No features show suspicious correlation (>0.5) with TARGET")
        print(f"  This is expected for a clean raw dataset — leakage comes from HOW we process it")

    if save_plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        top15 = correlations.head(15)
        colors = ['#e74c3c' if c > 0.5 else '#f39c12' if c > 0.3 else '#3498db'
                  for c in top15.values]
        ax.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Suspicious (>0.5)')
        ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='High (>0.3)')
        ax.set_xlabel('|Correlation with TARGET|')
        ax.set_title('Feature Correlation with TARGET\n(Leakage Hunt)', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig('outputs/03_leakage_hunt_correlations.png', dpi=150)
        plt.close()
        print(f"\n  [SAVED] outputs/03_leakage_hunt_correlations.png")


def _analyze_key_features(df: pd.DataFrame, save_plots: bool) -> None:
    """
    Compares distributions of key features between defaulters and non-defaulters.

    This tells us: do defaulters look different from people who repay?
    If yes — our model has real signal to learn from.
    If no — the features are useless and we have a hard problem.
    """

    print("\n── KEY FEATURE DISTRIBUTIONS ────────────────────────────")

    
    key_features = [
        'AMT_CREDIT',        
        'AMT_INCOME_TOTAL',  
        'AMT_ANNUITY',       
        'DAYS_BIRTH',        
        'DAYS_EMPLOYED',     
        'EXT_SOURCE_2',      
    ]

    
    available = [f for f in key_features if f in df.columns]

    print(f"  Comparing defaulters vs non-defaulters across {len(available)} key features:")

    defaulters = df[df['TARGET'] == 1]
    repayers   = df[df['TARGET'] == 0]

    for feat in available:
        d_median = defaulters[feat].median()
        r_median = repayers[feat].median()
        print(f"\n  {feat}:")
        print(f"    Defaulters median : {d_median:>12.2f}")
        print(f"    Repayers  median  : {r_median:>12.2f}")

    if save_plots and len(available) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, feat in enumerate(available[:6]):
            ax = axes[i]
            # Plot distribution for repayers
            repayers[feat].dropna().hist(
                bins=50, ax=ax, alpha=0.6, color='#2ecc71', label='Repaid (0)', density=True)
            # Plot distribution for defaulters
            defaulters[feat].dropna().hist(
                bins=50, ax=ax, alpha=0.6, color='#e74c3c', label='Defaulted (1)', density=True)
            ax.set_title(feat, fontweight='bold', fontsize=9)
            ax.legend(fontsize=7)
            ax.set_ylabel('Density')

        plt.suptitle('Key Feature Distributions: Defaulters vs Repayers',
                     fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/04_feature_distributions.png', dpi=150)
        plt.close()
        print(f"\n  [SAVED] outputs/04_feature_distributions.png")



if __name__ == "__main__":
    data_path = os.path.join("data", "application_train.csv")
    df = load_data(data_path)
    run_eda(df, save_plots=True)