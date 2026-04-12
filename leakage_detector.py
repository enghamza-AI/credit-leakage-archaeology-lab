import pandas as pd
import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LeakageDetector:
    def __init__(self, df, target_col='TARGET', id_col=None, group_col=None):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        self.df = df.copy()
        self.target_col = target_col
        self.id_col = id_col
        self.group_col = group_col
        self.results = {}
        print(f"[LeakageDetector] {df.shape[0]:,} rows x {df.shape[1]} cols | target={target_col}")

    def run_full_scan(self):
        print(f"\n[LeakageDetector] Scanning...\n")
        self.results = {
            'sin_1': self._check_sin1(),
            'sin_2': self._check_sin2(),
            'sin_3': self._check_sin3(),
            'sin_4': self._check_sin4(),
            'sin_5': self._check_sin5(),
        }
        self.results['score'] = self._compute_score()
        self.results['level'] = self._score_to_level(self.results['score'])
        return self.results

    def _check_sin1(self):
        cat_cols = [c for c in self.df.select_dtypes(include='object').columns if c != self.target_col]
        at_risk = [c for c in cat_cols if self.df[c].nunique() < 100]
        risk = 'CLEAN' if not at_risk else 'LOW' if len(at_risk)<=2 else 'MEDIUM' if len(at_risk)<=5 else 'HIGH'
        return {'name':'Sin 1: Target Encoding', 'risk':risk,
                'message':f"{len(at_risk)} categorical col(s) at risk of target encoding leak.",
                'fix':'Encode AFTER split or use Pipeline with TargetEncoder.'}

    def _check_sin2(self):
        num_cols = [c for c in self.df.select_dtypes(include=np.number).columns
                    if c not in [self.target_col, self.id_col or '']]
        if not num_cols:
            return {'name':'Sin 2: Feature From Target','risk':'CLEAN','message':'No numeric features.','fix':'N/A'}
        corrs = self.df[num_cols].corrwith(self.df[self.target_col]).abs().dropna().sort_values(ascending=False)
        critical = corrs[corrs > 0.8]
        high     = corrs[(corrs > 0.5) & (corrs <= 0.8)]
        risk = 'CRITICAL' if len(critical)>0 else 'HIGH' if len(high)>0 else 'MEDIUM' if len(corrs[corrs>0.3])>0 else 'CLEAN'
        flagged = corrs[corrs>0.5].head(5).round(4).to_dict()
        return {'name':'Sin 2: Feature From Target','risk':risk,
                'message':f"{len(critical)} critical (>0.8), {len(high)} high (>0.5) correlation features.",
                'flagged':flagged,'fix':"Ask: was this column computed using outcome data? If yes → drop it."}

    def _check_sin3(self):
        time_cols = [c for c in self.df.columns
                     if any(k in c.lower() for k in ['date','day','year','month','time','period'])
                     and c not in [self.target_col, self.id_col or '']]
        risk = 'MEDIUM' if time_cols else 'LOW'
        return {'name':'Sin 3: Timestamp Scaling','risk':risk,
                'message':f"{len(time_cols)} time-related column(s) found. Temporal ordering may matter.",
                'time_cols':time_cols,'fix':'Always use sklearn Pipeline. Never fit scaler on full dataset.'}

    def _check_sin4(self):
        col = self.group_col or self.id_col
        if not col or col not in self.df.columns:
            return {'name':'Sin 4: Group Overlap','risk':'UNKNOWN',
                    'message':'No group/ID column specified.','fix':'Specify id_col when creating detector.'}
        counts = self.df[col].value_counts()
        multi = (counts > 1).sum()
        total = len(counts)
        pct = multi/total
        risk = 'CLEAN' if multi==0 else 'LOW' if pct<0.05 else 'MEDIUM' if pct<0.20 else 'HIGH'
        return {'name':'Sin 4: Group Overlap','risk':risk,
                'message':f"{multi:,} of {total:,} groups ({pct*100:.1f}%) have multiple rows.",
                'fix':'Use GroupKFold — all rows from same group stay in one split.'}

    def _check_sin5(self):
        exact = int(self.df.duplicated().sum())
        id_dupes = 0
        if self.id_col and self.id_col in self.df.columns:
            id_dupes = int((self.df[self.id_col].value_counts() > 1).sum())
        pct = exact/len(self.df)*100
        risk = 'CLEAN' if exact==0 else 'LOW' if pct<1 else 'MEDIUM' if pct<5 else 'HIGH'
        return {'name':'Sin 5: Duplicate Rows','risk':risk,
                'message':f"{exact:,} exact duplicate rows ({pct:.2f}%). {id_dupes:,} duplicate IDs.",
                'exact_dupes':exact,'id_dupes':id_dupes,
                'fix':'Call df.drop_duplicates() BEFORE splitting. clean.py does this automatically.'}

    def _compute_score(self):
        weights = {'CRITICAL':25,'HIGH':15,'MEDIUM':8,'LOW':3,'CLEAN':0,'UNKNOWN':5}
        return min(sum(weights.get(v['risk'],0) for k,v in self.results.items()
                       if k not in ['score','level'] and isinstance(v,dict) and 'risk' in v), 100)

    def _score_to_level(self, score):
        return 'CLEAN' if score==0 else 'LOW' if score<=10 else 'MEDIUM' if score<=25 else 'HIGH' if score<=45 else 'CRITICAL'

    def print_report(self):
        icons = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢','CLEAN':'✅','UNKNOWN':'⚪'}
        print("\n" + "="*65)
        print("  LEAKAGE DETECTOR — SCAN REPORT")
        print("="*65)
        for key, r in self.results.items():
            if key in ['score','level']: continue
            if not isinstance(r, dict) or 'risk' not in r: continue
            icon = icons.get(r['risk'], '?')
            print(f"\n  {r['name']}")
            print(f"  Risk   : {icon} {r['risk']}")
            print(f"  Finding: {r['message']}")
            print(f"  Fix    : {r['fix']}")
            if 'flagged' in r and r['flagged']:
                print(f"  Flagged features:")
                for feat, corr in list(r['flagged'].items())[:3]:
                    print(f"    {feat:<35} corr={corr:.4f}")
        score = self.results['score']
        level = self.results['level']
        print(f"\n{'='*65}")
        print(f"  OVERALL SCORE : {score}/100")
        print(f"  OVERALL RISK  : {icons.get(level,'?')} {level}")
        print(f"{'='*65}")
        advice = {
            'CLEAN':'✅ Dataset passed all checks.',
            'LOW'  :'🟢 Low risk. Address before final deployment.',
            'MEDIUM':'🟡 Medium risk. Review flagged items before training.',
            'HIGH'  :'🟠 High risk. Fix issues before training.',
            'CRITICAL':'🔴 STOP. Fix all critical issues before proceeding.'
        }
        print(f"\n  {advice.get(level, '')}\n")

    def save_chart(self, path='outputs/leakage_report.png'):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        colors = {'CRITICAL':'#c0392b','HIGH':'#e67e22','MEDIUM':'#f1c40f',
                  'LOW':'#2ecc71','CLEAN':'#27ae60','UNKNOWN':'#95a5a6'}
        sev_map = {'CRITICAL':5,'HIGH':4,'MEDIUM':3,'LOW':2,'CLEAN':1,'UNKNOWN':1.5}
        sin_keys = [k for k in self.results if k not in ['score','level']]
        names  = [f"Sin {i+1}" for i in range(len(sin_keys))]
        risks  = [self.results[k]['risk'] for k in sin_keys]
        sevs   = [sev_map[r] for r in risks]
        cols   = [colors[r] for r in risks]

        fig, axes = plt.subplots(1, 2, figsize=(13,5))
        bars = axes[0].bar(names, sevs, color=cols, edgecolor='black')
        axes[0].set_yticks([1,2,3,4,5])
        axes[0].set_yticklabels(['CLEAN','LOW','MEDIUM','HIGH','CRITICAL'])
        axes[0].set_title('Leakage Risk Per Sin', fontweight='bold')
        for bar, risk in zip(bars, risks):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                         risk, ha='center', fontsize=8, fontweight='bold')

        score = self.results['score']
        level = self.results['level']
        c = colors.get(level,'#95a5a6')
        axes[1].barh(['Risk'], [score], color=c, edgecolor='black')
        axes[1].barh(['Risk'], [100-score], left=[score], color='#ecf0f1', edgecolor='black')
        axes[1].set_xlim(0,100)
        axes[1].text(score/2, 0, f'{score}/100', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='white' if score>20 else 'black')
        axes[1].set_title(f'Overall: {level} ({score}/100)', fontweight='bold', color=c)
        axes[1].set_yticks([])

        plt.suptitle('LeakageDetector — Automated Report', fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(path, dpi=150); plt.close()
        print(f"  [SAVED] {path}")

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from load_data import load_data
    df = load_data(os.path.join("data", "application_train.csv"))
    detector = LeakageDetector(df, target_col='TARGET', id_col='SK_ID_CURR', group_col='SK_ID_CURR')
    report = detector.run_full_scan()
    detector.print_report()
    os.makedirs("outputs", exist_ok=True)
    detector.save_chart('outputs/leakage_report.png')