
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

HONEST_COLOR = '#2ecc71'   
LEAKY_COLOR  = '#e74c3c'   
NEUTRAL_COLOR = '#3498db'  
BG_COLOR     = '#0f1520'   
TEXT_COLOR   = '#f0f4ff'   


def plot_auc_comparison(sin_results: list, save: bool = True) -> plt.Figure:
   

    os.makedirs("outputs", exist_ok=True)

    n = len(sin_results)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a2240')

   
    ax = axes[0]
    ax.set_facecolor('#141b2e')

    honest_aucs  = [r['honest_auc'] for r in sin_results]
    leaky_aucs   = [r['leaky_auc']  for r in sin_results]
    sin_labels   = [f"Sin {r['sin']}" for r in sin_results]

    bars_h = ax.bar(x - width/2, honest_aucs, width,
                    label='Honest AUC', color=HONEST_COLOR, edgecolor='white', linewidth=0.5)
    bars_l = ax.bar(x + width/2, leaky_aucs, width,
                    label='Leaky AUC',  color=LEAKY_COLOR,  edgecolor='white', linewidth=0.5)

   
    for bar in bars_h:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7, color=TEXT_COLOR, fontweight='bold')
    for bar in bars_l:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7, color=TEXT_COLOR, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(sin_labels, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('AUC Score', color=TEXT_COLOR)
    ax.set_title('Honest vs Leaky AUC — All 5 Sins', color=TEXT_COLOR, fontweight='bold', fontsize=11)
    ax.legend(facecolor='#1a2240', labelcolor=TEXT_COLOR, fontsize=9)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='Random baseline')
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3a5e')

   
    ax2 = axes[1]
    ax2.set_facecolor('#141b2e')

    inflations = [r['inflation'] for r in sin_results]
    colors_inf = [LEAKY_COLOR if inf > 0.05 else '#f39c12' if inf > 0.02 else HONEST_COLOR
                  for inf in inflations]
    sin_names  = [r['name'] for r in sin_results]

    bars_inf = ax2.barh(sin_names, inflations, color=colors_inf, edgecolor='white', linewidth=0.5)

    for bar, inf in zip(bars_inf, inflations):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'+{inf:.4f}', va='center', fontsize=9, color=TEXT_COLOR, fontweight='bold')

    ax2.set_xlabel('AUC Inflation (Leaky - Honest)', color=TEXT_COLOR)
    ax2.set_title('Leakage Inflation Per Sin\n(How much each sin inflates your score)',
                  color=TEXT_COLOR, fontweight='bold', fontsize=11)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.axvline(x=0.05, color='orange', linestyle='--', alpha=0.6, label='Warning (>0.05)')
    ax2.legend(facecolor='#1a2240', labelcolor=TEXT_COLOR, fontsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#2a3a5e')

    plt.suptitle('Leakage Archaeology Lab — AUC Inflation Summary',
                 color='#C9A84C', fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        plt.savefig('outputs/master_comparison.png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print("[SAVED] outputs/master_comparison.png")

    return fig


def plot_sin_detail(sin_num: int, honest_auc: float, leaky_auc: float,
                    sin_name: str, save: bool = True) -> plt.Figure:
   

    os.makedirs("outputs", exist_ok=True)

    inflation = leaky_auc - honest_auc
    sev_color = '#e74c3c' if inflation > 0.05 else '#f39c12' if inflation > 0.02 else '#2ecc71'
    sev_label = 'HIGH' if inflation > 0.05 else 'MODERATE' if inflation > 0.02 else 'LOW'

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#1a2240')

   
    ax = axes[0]
    ax.set_facecolor('#141b2e')
    bars = ax.bar(['Honest AUC', 'Leaky AUC'], [honest_auc, leaky_auc],
                  color=[HONEST_COLOR, LEAKY_COLOR], edgecolor='white', linewidth=0.6, width=0.5)
    for bar, val in zip(bars, [honest_auc, leaky_auc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{val:.4f}', ha='center', color=TEXT_COLOR, fontweight='bold', fontsize=12)
    ax.set_ylim(0.5, min(leaky_auc + 0.12, 1.0))
    ax.set_ylabel('AUC Score', color=TEXT_COLOR)
    ax.set_title(f'Sin {sin_num}: {sin_name}', color=TEXT_COLOR, fontweight='bold')
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3a5e')

    
    ax2 = axes[1]
    ax2.set_facecolor('#141b2e')
    ax2.set_xlim(0, 0.3)
    ax2.set_ylim(0, 1)

    
    ax2.barh([0.5], [inflation], height=0.3, color=sev_color, edgecolor='white')
    ax2.barh([0.5], [0.3 - inflation], left=[inflation], height=0.3,
             color='#2a3a5e', edgecolor='white')

    ax2.text(0.15, 0.75, f'AUC Inflation', ha='center', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax2.text(0.15, 0.5, f'+{inflation:.4f}', ha='center', va='center',
             color='white', fontsize=18, fontweight='bold')
    ax2.text(0.15, 0.25, f'Severity: {sev_label}', ha='center',
             color=sev_color, fontsize=11, fontweight='bold')
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title('Inflation Gauge', color=TEXT_COLOR, fontweight='bold')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#2a3a5e')

    plt.tight_layout()

    if save:
        path = f'outputs/sin_detail_{sin_num:02d}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[SAVED] {path}")

    return fig


def plot_detector_radar(detector_results: dict, save: bool = True) -> plt.Figure:
   

    os.makedirs("outputs", exist_ok=True)

    risk_to_score = {'CRITICAL': 5, 'HIGH': 4, 'MEDIUM': 3, 'LOW': 2, 'CLEAN': 1, 'UNKNOWN': 1.5}

    sin_keys   = [k for k in detector_results if k not in ['score', 'level'] and isinstance(detector_results[k], dict)]
    categories = [detector_results[k]['name'].replace('Sin ', 'Sin\n') for k in sin_keys]
    values     = [risk_to_score.get(detector_results[k]['risk'], 1) for k in sin_keys]

   
    categories += [categories[0]]
    values     += [values[0]]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a2240')

   
    ax = fig.add_subplot(121, polar=True)
    ax.set_facecolor('#141b2e')

    ax.plot(angles, values, color=LEAKY_COLOR, linewidth=2)
    ax.fill(angles, values, color=LEAKY_COLOR, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('\n', ' ') for c in categories[:-1]],
                       color=TEXT_COLOR, fontsize=8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['CLEAN', 'LOW', 'MED', 'HIGH', 'CRIT'], color='gray', fontsize=7)
    ax.set_ylim(0, 5)
    ax.set_title('Risk Radar\n(dataset fingerprint)', color=TEXT_COLOR,
                 fontweight='bold', pad=15)
    ax.grid(color='#2a3a5e', linewidth=0.5)
    ax.tick_params(colors=TEXT_COLOR)

   
    ax2 = axes[1]
    ax2.set_facecolor('#141b2e')

    score = detector_results.get('score', 0)
    level = detector_results.get('level', 'UNKNOWN')
    level_color = {'CRITICAL':'#e74c3c','HIGH':'#e67e22','MEDIUM':'#f1c40f',
                   'LOW':'#2ecc71','CLEAN':'#27ae60','UNKNOWN':'#95a5a6'}.get(level, '#95a5a6')

    
    sin_names_short = [f"Sin {i+1}" for i in range(len(sin_keys))]
    risk_scores     = values[:-1]
    colors_bars     = ['#e74c3c' if s>=4 else '#f39c12' if s>=3 else '#2ecc71' for s in risk_scores]

    bars = ax2.bar(sin_names_short, risk_scores, color=colors_bars, edgecolor='white', linewidth=0.5)
    ax2.set_yticks([1,2,3,4,5])
    ax2.set_yticklabels(['CLEAN','LOW','MED','HIGH','CRIT'], color=TEXT_COLOR, fontsize=9)
    ax2.set_ylim(0, 5.5)
    ax2.set_title(f'Overall Score: {score}/100 — {level}',
                  color=level_color, fontweight='bold', fontsize=11)
    ax2.tick_params(colors=TEXT_COLOR)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#2a3a5e')

    for bar, s in zip(bars, risk_scores):
        labels = {5:'CRIT',4:'HIGH',3:'MED',2:'LOW',1:'CLEAN'}
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 labels.get(int(s),'?'), ha='center', color=TEXT_COLOR, fontsize=8, fontweight='bold')

    plt.suptitle('LeakageDetector — Dataset Risk Profile',
                 color='#C9A84C', fontweight='bold', fontsize=13)
    plt.tight_layout()

    if save:
        plt.savefig('outputs/detector_radar.png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print("[SAVED] outputs/detector_radar.png")

    return fig


def plot_auc_explainer(save: bool = True) -> plt.Figure:
   

    os.makedirs("outputs", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#1a2240')

    
    ax = axes[0]
    ax.set_facecolor('#141b2e')

    fpr_random = np.linspace(0, 1, 100)
    tpr_random = fpr_random

    
    fpr_good = np.linspace(0, 1, 100)
    tpr_good = 1 - (1 - fpr_good)**3

    
    fpr_leaky = np.linspace(0, 1, 100)
    tpr_leaky = 1 - (1 - fpr_leaky)**6

    ax.plot(fpr_random, tpr_random, color='gray', linestyle='--', label='Random (AUC=0.50)')
    ax.plot(fpr_good,   tpr_good,   color=HONEST_COLOR, linewidth=2, label='Honest model (AUC≈0.72)')
    ax.plot(fpr_leaky,  tpr_leaky,  color=LEAKY_COLOR,  linewidth=2, label='Leaky model  (AUC≈0.91)')
    ax.fill_between(fpr_good, tpr_good, fpr_random, alpha=0.1, color=HONEST_COLOR)
    ax.fill_between(fpr_leaky, tpr_leaky, fpr_random, alpha=0.1, color=LEAKY_COLOR)

    ax.set_xlabel('False Positive Rate', color=TEXT_COLOR)
    ax.set_ylabel('True Positive Rate', color=TEXT_COLOR)
    ax.set_title('ROC Curve — Honest vs Leaky Model\n(AUC = area under these curves)',
                 color=TEXT_COLOR, fontweight='bold')
    ax.legend(facecolor='#1a2240', labelcolor=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3a5e')

   
    ax2 = axes[1]
    ax2.set_facecolor('#141b2e')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title('AUC Interpretation Guide', color=TEXT_COLOR, fontweight='bold')

    guide = [
        (0.95, 1.00, '#e74c3c', '0.95-1.00   Almost certainly leaking'),
        (0.90, 0.95, '#e67e22', '0.90-0.95   Very suspicious — investigate'),
        (0.80, 0.90, '#2ecc71', '0.80-0.90   Strong — genuinely good model'),
        (0.70, 0.80, '#3498db', '0.70-0.80   Decent — learning real signal'),
        (0.60, 0.70, '#9b59b6', '0.60-0.70   Weak — model struggling'),
        (0.50, 0.60, 'gray',    '0.50-0.60   Barely better than random'),
    ]
    for i, (lo, hi, color, label) in enumerate(guide):
        y = 0.88 - i * 0.14
        ax2.barh([y], [hi - lo], left=[lo], height=0.10, color=color, alpha=0.8)
        ax2.text(0.02, y, label, va='center', color=TEXT_COLOR, fontsize=8.5)

    for spine in ax2.spines.values():
        spine.set_edgecolor('#2a3a5e')

    plt.suptitle('Understanding AUC', color='#C9A84C', fontweight='bold', fontsize=13)
    plt.tight_layout()

    if save:
        plt.savefig('outputs/auc_explainer.png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print("[SAVED] outputs/auc_explainer.png")

    return fig


def generate_all_charts(sin_results: list, detector_results: dict = None) -> None:
    

    print("[visualize] Generating all charts...")
    plot_auc_explainer(save=True)
    plot_auc_comparison(sin_results, save=True)
    for r in sin_results:
        plot_sin_detail(r['sin'], r['honest_auc'], r['leaky_auc'], r['name'], save=True)
    if detector_results:
        plot_detector_radar(detector_results, save=True)
    print("[visualize] All charts saved to outputs/")



if __name__ == "__main__":
    
    demo_results = [
        {'sin':1,'name':'Target Encoding',       'honest_auc':0.710,'leaky_auc':0.748,'inflation':0.038},
        {'sin':2,'name':'Feature From Target',   'honest_auc':0.710,'leaky_auc':0.890,'inflation':0.180},
        {'sin':3,'name':'Timestamp Scaling',     'honest_auc':0.708,'leaky_auc':0.725,'inflation':0.017},
        {'sin':4,'name':'Group Overlap',         'honest_auc':0.695,'leaky_auc':0.720,'inflation':0.025},
        {'sin':5,'name':'Duplicate ID',          'honest_auc':0.710,'leaky_auc':0.740,'inflation':0.030},
    ]
    demo_detector = {
        'sin_1':{'name':'Sin 1: Target Encoding',  'risk':'MEDIUM'},
        'sin_2':{'name':'Sin 2: Feature From Target','risk':'CLEAN'},
        'sin_3':{'name':'Sin 3: Timestamp Scaling', 'risk':'LOW'},
        'sin_4':{'name':'Sin 4: Group Overlap',     'risk':'CLEAN'},
        'sin_5':{'name':'Sin 5: Duplicate Rows',    'risk':'CLEAN'},
        'score': 8, 'level': 'LOW'
    }
    generate_all_charts(demo_results, demo_detector)
    print("\n[DONE] All charts generated. Check outputs/ folder.")