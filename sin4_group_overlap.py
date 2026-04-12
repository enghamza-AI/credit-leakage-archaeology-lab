import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data
from clean import clean_data, get_features_and_target
from split import random_split
from train import train_and_evaluate, compare_honest_vs_leaky

def run_sin_04(df, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("  SIN 4: GROUP OVERLAP LEAKAGE")
        print("="*60)

    df_clean = clean_data(df, verbose=False)
    X, y = get_features_and_target(df_clean)
    borrower_ids = df_clean['SK_ID_CURR'].reset_index(drop=True) if 'SK_ID_CURR' in df_clean.columns else pd.Series(range(len(X)))
    X = X.fillna(X.median(numeric_only=True)).reset_index(drop=True)
    y = y.reset_index(drop=True)

    
    X_sim, y_sim, groups_sim = _simulate_overlap(X, y, borrower_ids, verbose)

    
    if verbose: print(f"\n[LEAKY] Random split — borrowers can appear in train AND test")
    X_tr_l, X_te_l, y_tr_l, y_te_l = random_split(X_sim, y_sim, verbose=False)
    train_g = set(groups_sim.iloc[X_tr_l.index])
    test_g  = set(groups_sim.iloc[X_te_l.index])
    overlap = len(train_g & test_g)
    if verbose: print(f"  Borrowers in BOTH train and test: {overlap:,} <- the sin")
    leaky = train_and_evaluate(X_tr_l, X_te_l, y_tr_l, y_te_l, label="SIN 4 LEAKY", verbose=verbose)

    
    if verbose: print(f"\n[HONEST] Group K-Fold — zero borrower overlap guaranteed")
    honest_auc = _group_kfold_eval(X_sim, y_sim, groups_sim, verbose)
    if verbose: print(f"  Honest AUC (avg 5 folds): {honest_auc:.4f}")

    compare_honest_vs_leaky({'auc':honest_auc}, leaky, "SIN 4: GROUP OVERLAP")
    os.makedirs("outputs", exist_ok=True)
    _save_chart(honest_auc, leaky['auc'], overlap, len(train_g))
    return {'sin':4, 'honest_auc':honest_auc, 'leaky_auc':leaky['auc'],
            'inflation':leaky['auc']-honest_auc}

def _simulate_overlap(X, y, ids, verbose):
    np.random.seed(42)
    unique_ids = ids.unique()
    to_dup = np.random.choice(unique_ids, size=int(len(unique_ids)*0.30), replace=False)
    mask = ids.isin(to_dup)
    X_d = X[mask].copy(); y_d = y[mask].copy(); g_d = ids[mask].copy()
    noise = np.random.normal(0, 0.05, X_d.shape)
    X_d = X_d + noise
    X_s = pd.concat([X, X_d], ignore_index=True)
    y_s = pd.concat([y, y_d], ignore_index=True)
    g_s = pd.concat([ids, g_d], ignore_index=True)
    if verbose:
        print(f"\n  Simulated {len(X_d):,} duplicate borrower rows (30% of borrowers appear twice)")
        print(f"  Total rows: {len(X_s):,}")
    return X_s, y_s, g_s

def _group_kfold_eval(X, y, groups, verbose):
    gkf = GroupKFold(n_splits=5)
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler()),
                     ('m',   LogisticRegression(max_iter=1000, C=0.1, random_state=42))])
    aucs = []
    for fold, (tr_i, te_i) in enumerate(gkf.split(X, y, groups)):
        pipe.fit(X.iloc[tr_i], y.iloc[tr_i])
        auc = roc_auc_score(y.iloc[te_i], pipe.predict_proba(X.iloc[te_i])[:,1])
        aucs.append(auc)
        ov = len(set(groups.iloc[tr_i]) & set(groups.iloc[te_i]))
        if verbose: print(f"  Fold {fold+1}: AUC={auc:.4f} borrower_overlap={ov} {'✅' if ov==0 else '❌'}")
    return np.mean(aucs)

def _save_chart(honest, leaky, overlap, total_groups):
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    bars = axes[0].bar(['Honest\n(Group KFold)','Leaky\n(Random — Sin 4)'],
                       [honest, leaky], color=['#2ecc71','#e74c3c'], edgecolor='black')
    for bar, val in zip(bars, [honest, leaky]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                     f'{val:.4f}', ha='center', fontweight='bold')
    axes[0].set_ylim(0.5, min(leaky+0.08,1.0))
    axes[0].set_title(f'AUC Inflation: +{leaky-honest:.4f}', fontweight='bold')
    non_ov = total_groups - overlap
    axes[1].pie([non_ov, overlap],
                labels=[f'Train only\n{non_ov:,}', f'BOTH sets\n{overlap:,} <- SIN'],
                colors=['#3498db','#e74c3c'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Borrower Overlap in Random Split', fontweight='bold')
    plt.suptitle('Sin 4: Group Overlap — Same Borrower in Train and Test', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sin_04.png', dpi=150); plt.close()
    print("  [SAVED] outputs/sin_04.png")

if __name__ == "__main__":
    df = load_data(os.path.join("data", "application_train.csv"))
    r = run_sin_04(df)
    print(f"\n[SIN 4] Inflation: +{r['inflation']:.4f}")