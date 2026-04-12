import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data
from clean import clean_data, get_features_and_target
from train import compare_honest_vs_leaky

def run_sin_03(df, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("  SIN 3: TIMESTAMP-CONTAMINATED SCALING")
        print("="*60)

    df_clean = clean_data(df, verbose=False)
    X, y = get_features_and_target(df_clean)
    X = X.fillna(X.median(numeric_only=True))

    
    split_idx = int(len(X) * 0.80)
    X_train = X.iloc[:split_idx]; X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]; y_test = y.iloc[split_idx:]

    if verbose:
        print(f"\n  Train (historical): {len(X_train):,} rows")
        print(f"  Test  (future)    : {len(X_test):,} rows")

    
    if verbose: print(f"\n[LEAKY] Scaler fitted on full dataset (sees test data)")
    imp = SimpleImputer(strategy='median')
    X_all_imp = imp.fit_transform(X)  # fits on ALL rows
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_imp)  # learns stats from ALL rows
    X_tr_l = X_all_scaled[:split_idx]; X_te_l = X_all_scaled[split_idx:]
    model_l = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    model_l.fit(X_tr_l, y_train)
    leaky_auc = roc_auc_score(y_test, model_l.predict_proba(X_te_l)[:,1])
    if verbose: print(f"  Leaky AUC: {leaky_auc:.4f}")

    
    if verbose: print(f"\n[HONEST] Pipeline — scaler fitted on train only")
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler()),
                     ('m',   LogisticRegression(max_iter=1000, C=0.1, random_state=42))])
    pipe.fit(X_train, y_train)
    honest_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    if verbose: print(f"  Honest AUC: {honest_auc:.4f}")

    compare_honest_vs_leaky({'auc':honest_auc}, {'auc':leaky_auc}, "SIN 3: TIMESTAMP SCALING LEAK")

    if verbose:
        print(f"\n  THE FIX: Always use sklearn Pipeline.")
        print(f"  pipeline.fit(X_train) -> scaler learns from train ONLY")
        print(f"  pipeline.transform(X_test) -> applies train stats to test")

    os.makedirs("outputs", exist_ok=True)
    _save_chart(honest_auc, leaky_auc, split_idx, len(X))
    return {'sin':3, 'honest_auc':honest_auc, 'leaky_auc':leaky_auc,
            'inflation':leaky_auc-honest_auc}

def _save_chart(honest, leaky, split_idx, total):
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    bars = axes[0].bar(['Honest\n(Pipeline)','Leaky\n(scaled all — Sin 3)'],
                       [honest, leaky], color=['#2ecc71','#e74c3c'], edgecolor='black')
    for bar, val in zip(bars, [honest, leaky]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                     f'{val:.4f}', ha='center', fontweight='bold')
    axes[0].set_ylim(0.5, min(leaky+0.08,1.0))
    axes[0].set_title(f'AUC Inflation: +{leaky-honest:.4f}', fontweight='bold')
    tr = split_idx/total; te = 1-tr
    axes[1].barh(['Split'], [tr], color='#3498db', label=f'Train {tr*100:.0f}%')
    axes[1].barh(['Split'], [te], left=[tr], color='#e74c3c', label=f'Test {te*100:.0f}%')
    axes[1].axvline(x=tr, color='black', linewidth=2, linestyle='--')
    axes[1].text(tr/2, 0, 'TRAIN\n(Past)', ha='center', va='center', color='white', fontweight='bold')
    axes[1].text(tr+te/2, 0, 'TEST\n(Future)', ha='center', va='center', color='white', fontweight='bold')
    axes[1].set_title('Temporal Split\nScaler must learn from TRAIN only', fontweight='bold')
    axes[1].set_yticks([]); axes[1].legend()
    plt.suptitle('Sin 3: Timestamp-Contaminated Scaling', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sin_03.png', dpi=150); plt.close()
    print("  [SAVED] outputs/sin_03.png")

if __name__ == "__main__":
    df = load_data(os.path.join("data", "application_train.csv"))
    r = run_sin_03(df)
    print(f"\n[SIN 3] Inflation: +{r['inflation']:.4f}")