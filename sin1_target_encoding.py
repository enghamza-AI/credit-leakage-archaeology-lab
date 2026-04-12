import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data
from clean import clean_data
from split import random_split
from train import train_and_evaluate, compare_honest_vs_leaky

def run_sin_01(df, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("  SIN 1: TARGET ENCODING LEAK")
        print("="*60)

    cat_col = 'NAME_EDUCATION_TYPE'
    num_cols = [c for c in ['AMT_CREDIT','AMT_INCOME_TOTAL','AMT_ANNUITY',
                'DAYS_BIRTH','DAYS_EMPLOYED','EXT_SOURCE_2','EXT_SOURCE_3'] if c in df.columns]
    cols = [cat_col, 'TARGET'] + num_cols
    data = df[cols].dropna(subset=[cat_col]).copy()

    if verbose:
        print(f"\n  Dataset: {len(data):,} rows")
        rates = data.groupby(cat_col)['TARGET'].mean()
        print(f"\n  Default rate by education:")
        for cat, rate in rates.items():
            print(f"    {cat:<35} {rate:.3f}")

    
    global_enc = data.groupby(cat_col)['TARGET'].mean()
    df_leaky = data.copy()
    df_leaky['EDU_ENCODED'] = df_leaky[cat_col].map(global_enc)
    df_leaky = df_leaky.drop(columns=[cat_col])
    X_l = df_leaky.drop(columns=['TARGET'])
    y   = df_leaky['TARGET']
    X_train_l, X_test_l, y_train_l, y_test_l = random_split(X_l, y, verbose=False)
    leaky = train_and_evaluate(X_train_l, X_test_l, y_train_l, y_test_l,
                                label="SIN 1 LEAKY", verbose=verbose)

    
    X_raw = data[num_cols + [cat_col]]
    y     = data['TARGET']
    X_tr, X_te, y_tr, y_te = random_split(X_raw, y, verbose=False)
    train_copy = X_tr.copy()
    train_copy['TARGET'] = y_tr.values
    train_enc = train_copy.groupby(cat_col)['TARGET'].mean()
    X_tr = X_tr.copy(); X_te = X_te.copy()
    X_tr['EDU_ENCODED'] = X_tr[cat_col].map(train_enc).fillna(y_tr.mean())
    X_te['EDU_ENCODED'] = X_te[cat_col].map(train_enc).fillna(y_tr.mean())
    X_tr = X_tr.drop(columns=[cat_col])
    X_te = X_te.drop(columns=[cat_col])
    honest = train_and_evaluate(X_tr, X_te, y_tr, y_te,
                                 label="SIN 1 HONEST", verbose=verbose)

    compare_honest_vs_leaky(honest, leaky, "SIN 1: TARGET ENCODING LEAK")
    os.makedirs("outputs", exist_ok=True)
    _save_chart(honest['auc'], leaky['auc'])
    return {'sin':1, 'honest_auc':honest['auc'], 'leaky_auc':leaky['auc'],
            'inflation':leaky['auc']-honest['auc']}

def _save_chart(honest, leaky):
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(['Honest','Leaky (Sin 1)'], [honest, leaky],
                  color=['#2ecc71','#e74c3c'], edgecolor='black')
    for bar, val in zip(bars, [honest, leaky]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{val:.4f}', ha='center', fontweight='bold')
    ax.set_ylim(0.5, min(leaky+0.1, 1.0))
    ax.set_ylabel('AUC'); ax.set_title(f'Sin 1: Target Encoding Leak\nInflation: +{leaky-honest:.4f}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sin_01.png', dpi=150); plt.close()
    print("  [SAVED] outputs/sin_01.png")

if __name__ == "__main__":
    df = load_data(os.path.join("data", "application_train.csv"))
    r = run_sin_01(df)
    print(f"\n[SIN 1] Inflation: +{r['inflation']:.4f}")