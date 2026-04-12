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

def run_sin_05(df, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("  SIN 5: DUPLICATE ID LEAKAGE")
        print("="*60)

    df_clean = clean_data(df, verbose=False)
    X, y = get_features_and_target(df_clean)
    X = X.fillna(X.median(numeric_only=True)).reset_index(drop=True)
    y = y.reset_index(drop=True)

    
    X_leaky, y_leaky = _inject_duplicates(X, y, fraction=0.08, verbose=verbose)

    
    if verbose: print(f"\n[LEAKY] Splitting WITHOUT removing duplicates")
    X_tr_l, X_te_l, y_tr_l, y_te_l = random_split(X_leaky, y_leaky, verbose=False)
    leaky = train_and_evaluate(X_tr_l, X_te_l, y_tr_l, y_te_l, label="SIN 5 LEAKY", verbose=verbose)

    
    if verbose: print(f"\n[HONEST] Remove duplicates, then split")
    combined = X_leaky.copy(); combined['__T__'] = y_leaky.values
    deduped = combined.drop_duplicates().reset_index(drop=True)
    y_h = deduped['__T__']; X_h = deduped.drop(columns=['__T__'])
    removed = len(combined) - len(deduped)
    if verbose: print(f"  Removed {removed:,} duplicate rows before splitting")
    X_tr_h, X_te_h, y_tr_h, y_te_h = random_split(X_h, y_h, verbose=False)
    honest = train_and_evaluate(X_tr_h, X_te_h, y_tr_h, y_te_h, label="SIN 5 HONEST", verbose=verbose)

    compare_honest_vs_leaky(honest, leaky, "SIN 5: DUPLICATE ID LEAKAGE")

    if verbose:
        print(f"\n  DETECTION — one line of code:")
        print(f"    n_dupes = df.duplicated().sum()")
        print(f"    if n_dupes > 0: df = df.drop_duplicates()")
        print(f"\n  clean.py already does this — it is protection built in.")

    os.makedirs("outputs", exist_ok=True)
    _save_chart(honest['auc'], leaky['auc'], len(X), len(X_leaky))
    return {'sin':5, 'honest_auc':honest['auc'], 'leaky_auc':leaky['auc'],
            'inflation':leaky['auc']-honest['auc']}

def _inject_duplicates(X, y, fraction, verbose):
    np.random.seed(42)
    n_dup = int(len(X) * fraction)
    idx = np.random.choice(len(X), size=n_dup, replace=False)
    X_d = X.iloc[idx].copy(); y_d = y.iloc[idx].copy()
    X_c = pd.concat([X, X_d], ignore_index=True)
    y_c = pd.concat([y, y_d], ignore_index=True)
    if verbose:
        print(f"\n  Injected {n_dup:,} duplicate rows ({fraction*100:.0f}% of data)")
        print(f"  Total rows now: {len(X_c):,}")
        found = X_c.duplicated().sum()
        print(f"  df.duplicated().sum() = {found:,}  <- one line catches this")
    return X_c, y_c

def _save_chart(honest, leaky, n_orig, n_leaky):
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    bars = axes[0].bar(['Honest\n(dedup first)','Leaky\n(kept dupes — Sin 5)'],
                       [honest, leaky], color=['#2ecc71','#e74c3c'], edgecolor='black')
    for bar, val in zip(bars, [honest, leaky]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                     f'{val:.4f}', ha='center', fontweight='bold')
    axes[0].set_ylim(0.5, min(leaky+0.08,1.0))
    axes[0].set_title(f'AUC Inflation: +{leaky-honest:.4f}', fontweight='bold')
    axes[1].bar(['Original Rows','Injected Dupes'], [n_orig, n_leaky-n_orig],
                color=['#3498db','#e74c3c'], edgecolor='black')
    axes[1].set_title('ETL Bug: Duplicate Rows Injected', fontweight='bold')
    axes[1].set_ylabel('Row Count')
    plt.suptitle('Sin 5: Duplicate ID Leakage', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sin_05.png', dpi=150); plt.close()
    print("  [SAVED] outputs/sin_05.png")

if __name__ == "__main__":
    df = load_data(os.path.join("data", "application_train.csv"))
    r = run_sin_05(df)
    print(f"\n[SIN 5] Inflation: +{r['inflation']:.4f}")