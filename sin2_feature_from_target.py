import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data
from clean import clean_data, get_features_and_target
from split import random_split
from train import train_and_evaluate, compare_honest_vs_leaky

def run_sin_02(df, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("  SIN 2: FEATURE DERIVED FROM TARGET")
        print("="*60)

    df_clean = clean_data(df, verbose=False)
    X, y = get_features_and_target(df_clean)
    X = X.fillna(X.median(numeric_only=True)).reset_index(drop=True)
    y = y.reset_index(drop=True)

    np.random.seed(42)
    n = len(y)
    
    leaky_feat = np.where(y==1,
        np.clip(np.random.normal(0.80, 0.15, n), 0, 1),
        np.clip(np.random.normal(0.20, 0.15, n), 0, 1))
    corr = np.corrcoef(leaky_feat, y.values)[0,1]
    if verbose:
        print(f"\n  Injecting SYNTHETIC_RISK_SCORE (simulates a post-outcome feature)")
        print(f"  Correlation with TARGET: {corr:.4f}  [normal range: 0.01-0.15]")

    
    X_leaky = X.copy()
    X_leaky['SYNTHETIC_RISK_SCORE'] = leaky_feat
    X_tr_l, X_te_l, y_tr_l, y_te_l = random_split(X_leaky, y, verbose=False)
    leaky = train_and_evaluate(X_tr_l, X_te_l, y_tr_l, y_te_l, label="SIN 2 LEAKY", verbose=verbose)

    
    X_tr_h, X_te_h, y_tr_h, y_te_h = random_split(X, y, verbose=False)
    honest = train_and_evaluate(X_tr_h, X_te_h, y_tr_h, y_te_h, label="SIN 2 HONEST", verbose=verbose)

    compare_honest_vs_leaky(honest, leaky, "SIN 2: FEATURE FROM TARGET")

    if verbose:
        print(f"\n  HOW TO DETECT: Check correlation of every feature with TARGET")
        print(f"  Flag anything > 0.5 and ask: was this column computed using outcome data?")

    os.makedirs("outputs", exist_ok=True)
    _save_chart(honest['auc'], leaky['auc'], leaky_feat, y)
    return {'sin':2, 'honest_auc':honest['auc'], 'leaky_auc':leaky['auc'],
            'inflation':leaky['auc']-honest['auc']}

def _save_chart(honest, leaky, leaky_feat, y):
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    bars = axes[0].bar(['Honest','Leaky (Sin 2)'], [honest, leaky],
                       color=['#2ecc71','#e74c3c'], edgecolor='black')
    for bar, val in zip(bars, [honest, leaky]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                     f'{val:.4f}', ha='center', fontweight='bold')
    axes[0].set_ylim(0.5, min(leaky+0.1,1.0))
    axes[0].set_title(f'AUC Inflation: +{leaky-honest:.4f}', fontweight='bold')
    axes[1].hist(leaky_feat[y==0], bins=40, alpha=0.6, color='#2ecc71', label='Repaid', density=True)
    axes[1].hist(leaky_feat[y==1], bins=40, alpha=0.6, color='#e74c3c', label='Defaulted', density=True)
    axes[1].set_title('Leaky Feature Distribution\n(separates classes — that is the sin)', fontweight='bold')
    axes[1].legend()
    plt.suptitle('Sin 2: Feature Derived From Target', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sin_02.png', dpi=150); plt.close()
    print("  [SAVED] outputs/sin_02.png")

if __name__ == "__main__":
    df = load_data(os.path.join("data", "application_train.csv"))
    r = run_sin_02(df)
    print(f"\n[SIN 2] Inflation: +{r['inflation']:.4f}")