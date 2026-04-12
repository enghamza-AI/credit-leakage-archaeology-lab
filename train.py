

import pandas as pd
import numpy as np
import os
import sys


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data
from clean import clean_data, get_features_and_target
from split import random_split


def build_honest_pipeline() -> Pipeline:


    pipeline = Pipeline(steps=[

       
        ('imputer', SimpleImputer(strategy='median')),

       
        ('scaler', StandardScaler()),

       
        ('model', LogisticRegression(
            max_iter=1000,
            C=0.1,
            solver='lbfgs',
            random_state=42
        ))
    ])

    return pipeline


def train_and_evaluate(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       label: str = "HONEST BASELINE",
                       verbose: bool = True) -> dict:
  

    
    pipeline = build_honest_pipeline()

    
    pipeline.fit(X_train, y_train)

   
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

 
    auc = roc_auc_score(y_test, y_pred_proba)

    results = {
        'label'        : label,
        'auc'          : auc,
        'n_train'      : len(X_train),
        'n_test'       : len(X_test),
        'default_rate' : y_test.mean(),
        'pipeline'     : pipeline
    }

    if verbose:
        _print_results(results)

    return results


def compare_honest_vs_leaky(honest_results: dict,
                             leaky_results: dict,
                             sin_name: str = "UNKNOWN SIN") -> None:
 

    honest_auc = honest_results['auc']
    leaky_auc  = leaky_results['auc']
    inflation  = leaky_auc - honest_auc

    print("\n" + "="*60)
    print(f"  RESULT: {sin_name}")
    print("="*60)
    print(f"  Honest AUC  : {honest_auc:.4f}  ← what you'd get in production")
    print(f"  Leaky  AUC  : {leaky_auc:.4f}  ← what the sin makes you THINK you have")
    print(f"  Inflation   : +{inflation:.4f}  ← the LIE")
    print()


    honest_bar = "█" * int(honest_auc * 40)
    leaky_bar  = "█" * int(leaky_auc * 40)
    print(f"  Honest [{honest_bar:<40}] {honest_auc:.3f}")
    print(f"  Leaky  [{leaky_bar:<40}] {leaky_auc:.3f}")
    print()

    
    if inflation > 0.15:
        severity = "CRITICAL"
        icon = "🔴"
    elif inflation > 0.08:
        severity = "HIGH"
        icon = "🟠"
    elif inflation > 0.03:
        severity = "MODERATE"
        icon = "🟡"
    else:
        severity = "LOW"
        icon = "🟢"

    print(f"  Leakage Severity: {icon} {severity}")
    print(f"\n  In production, this model would UNDERPERFORM expectations by {inflation:.1%}")
    print(f"  Real-world loans would default at rates your model never learned to catch.")
    print("="*60)


def _print_results(results: dict) -> None:
    

    print(f"\n── {results['label']} ──")
    print(f"  Train size    : {results['n_train']:,}")
    print(f"  Test size     : {results['n_test']:,}")
    print(f"  Default rate  : {results['default_rate']*100:.2f}%")
    print(f"  AUC Score     : {results['auc']:.4f}")

    
    auc = results['auc']
    if auc >= 0.90:
        interpretation = "  VERY HIGH — check for leakage"
    elif auc >= 0.80:
        interpretation = " STRONG — genuinely good"
    elif auc >= 0.70:
        interpretation = " DECENT — learning real signal"
    else:
        interpretation = "  WEAK — model struggling"

    print(f"  Interpretation: {interpretation}")



if __name__ == "__main__":
    data_path = os.path.join("data", "application_train.csv")

    print("[INFO] Loading and cleaning data...")
    df       = load_data(data_path)
    df_clean = clean_data(df, verbose=False)
    X, y     = get_features_and_target(df_clean)

    print(f"[INFO] Features: {X.shape[1]} columns")

    
    X_train, X_test, y_train, y_test = random_split(X, y, verbose=True)

    
    results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        label="HONEST BASELINE — No Leakage",
        verbose=True
    )

    print(f"\n[BASELINE ESTABLISHED]")
    print(f"  Honest AUC = {results['auc']:.4f}")
    print(f"  This is the number all sin files will compare against.")
    print(f"  When a sin inflates this — that is the lie we are measuring.")