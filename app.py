 


import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


st.set_page_config(
    page_title="Leakage Archaeology Lab",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #080C14; color: #F0F4FF; }
    .main-header {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg, #C9A84C, #F0D080);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #9AA5BC; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .sin-card {
        background: #0F1520; border: 1px solid #1A2240;
        border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0;
    }
    .metric-card {
        background: #141B2E; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .auc-honest { color: #2ecc71; font-size: 2rem; font-weight: 800; }
    .auc-leaky  { color: #e74c3c; font-size: 2rem; font-weight: 800; }
    .inflation  { color: #f39c12; font-size: 1.5rem; font-weight: 800; }
    .risk-critical { color: #e74c3c; font-weight: 700; }
    .risk-high     { color: #e67e22; font-weight: 700; }
    .risk-medium   { color: #f1c40f; font-weight: 700; }
    .risk-low      { color: #2ecc71; font-weight: 700; }
    .risk-clean    { color: #27ae60; font-weight: 700; }
    .tag {
        display: inline-block; background: #141B2E;
        border: 1px solid #7A5F20; color: #F0D080;
        border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; margin: 2px;
    }
    section[data-testid="stSidebar"] { background-color: #0F1520; }
    .stSelectbox label, .stSlider label { color: #9AA5BC; }
</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.markdown("## 🏦 Leakage Archaeology Lab")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 AUC Explainer",
        "🔍 Sin Explorer",
        "📈 Master Comparison",
        "🤖 Auto Detector"
    ])
    st.markdown("---")
    st.markdown("**Built by** [enghamza-AI](https://github.com/enghamza-AI)")
    st.markdown("**Dataset:** Home Credit Default Risk (350k loans)")
    st.markdown("**Stage:** 1 · Week 4 of Diamond AI Roadmap")
    st.markdown("---")
    st.markdown("**Tech Stack**")
    for tag in ['Python','Scikit-learn','Pandas','Matplotlib','Streamlit']:
        st.markdown(f'<span class="tag">{tag}</span>', unsafe_allow_html=True)



@st.cache_data
def load_dataset():
    
    try:
        from load_data import load_data
        df = load_data(os.path.join("data", "application_train.csv"))
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def get_sin_results():
    
    df, err = load_dataset()
    if err or df is None:
        
        return [
            {'sin':1,'name':'Target Encoding',      'honest_auc':0.710,'leaky_auc':0.748,'inflation':0.038},
            {'sin':2,'name':'Feature From Target',  'honest_auc':0.710,'leaky_auc':0.890,'inflation':0.180},
            {'sin':3,'name':'Timestamp Scaling',    'honest_auc':0.708,'leaky_auc':0.725,'inflation':0.017},
            {'sin':4,'name':'Group Overlap',        'honest_auc':0.695,'leaky_auc':0.720,'inflation':0.025},
            {'sin':5,'name':'Duplicate ID',         'honest_auc':0.710,'leaky_auc':0.740,'inflation':0.030},
        ], True  

    results = []
    try:
        from sin1_target_encoding   import run_sin_01
        from sin2_feature_from_target import run_sin_02
        from sin3_timestamp_scaling  import run_sin_03
        from sin4_group_overlap      import run_sin_04
        from sin5_duplicate_id       import run_sin_05

        results.append(run_sin_01(df, verbose=False))
        results.append(run_sin_02(df, verbose=False))
        results.append(run_sin_03(df, verbose=False))
        results.append(run_sin_04(df, verbose=False))
        results.append(run_sin_05(df, verbose=False))
        return results, False

    except Exception as e:
        st.error(f"Error running sins: {e}")
        return [], False



SIN_INFO = {
    1: {
        'name'      : 'Target Encoding Leak',
        'icon'      : '🎯',
        'story'     : 'You encode a category (education type) using default rates from the FULL dataset before splitting. The test set\'s outcomes quietly contaminate the training features.',
        'the_sin'   : 'Computing target encoding statistics BEFORE the train/test split.',
        'the_fix'   : 'Encode AFTER splitting. Use only training data to compute encoding statistics. Apply those training stats to the test set.',
        'detection' : 'Check if categorical columns were encoded before or after split.',
    },
    2: {
        'name'      : 'Feature Derived From Target',
        'icon'      : '🧬',
        'story'     : 'A feature in your dataset was calculated using final loan outcomes — data that only exists AFTER the loan ended. It looks like a legitimate feature but it already knows the answer.',
        'the_sin'   : 'Including a feature that contains post-outcome information.',
        'the_fix'   : 'Check every feature: was it available at prediction time? If it uses outcome data → drop it.',
        'detection' : 'Correlation of any feature with TARGET > 0.5 → investigate immediately.',
    },
    3: {
        'name'      : 'Timestamp-Contaminated Scaling',
        'icon'      : '⏱️',
        'story'     : 'You fit StandardScaler on the full dataset (train + test) before splitting. Future test data statistics contaminate how training data is scaled.',
        'the_sin'   : 'Fitting scaler on ALL data. Test statistics influence training feature scale.',
        'the_fix'   : 'Always use sklearn Pipeline. pipeline.fit(X_train) fits scaler on train only.',
        'detection' : 'Look for any scaler.fit(X_all) call outside of a Pipeline.',
    },
    4: {
        'name'      : 'Group Overlap',
        'icon'      : '👥',
        'story'     : 'The same borrower appears in both training and test. The model recognizes the person from training and "predicts" based on memory, not learned patterns.',
        'the_sin'   : 'Random split that ignores group membership — same entity in train and test.',
        'the_fix'   : 'Use GroupKFold. All rows from the same borrower go to either train OR test, never both.',
        'detection' : 'Check if any ID appears in both X_train and X_test after splitting.',
    },
    5: {
        'name'      : 'Duplicate ID Leakage',
        'icon'      : '📋',
        'story'     : 'An ETL bug caused 8% of rows to be duplicated. After random split, copy A is in training and copy B is in test. The model memorizes exact answers during training.',
        'the_sin'   : 'Identical rows split across train and test. Model memorizes test answers.',
        'the_fix'   : 'Call df.drop_duplicates() BEFORE splitting. One line prevents this entirely.',
        'detection' : 'df.duplicated().sum() — if > 0, fix before proceeding.',
    },
}



if page == "🏠 Overview":
    st.markdown('<p class="main-header">🏦 Leakage Archaeology Lab</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Excavating the 5 sins of credit ML — and building a detector that catches them automatically.</p>', unsafe_allow_html=True)

    
    st.info("**Core Truth:** A model's AUC score means nothing if your data preparation was wrong. This project finds, demonstrates, and detects the 5 most dangerous ways ML pipelines silently cheat.")

    st.markdown("---")

    
    st.markdown("### 📊 Dataset")
    df, err = load_dataset()

    if err:
        st.warning(f"⚠️ Dataset not found. Place `application_train.csv` in the `data/` folder.\n\nShowing demo mode with placeholder values.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", "307,511")
        c2.metric("Features", "120+")
        c3.metric("Default Rate", "~8.07%")
        c4.metric("Dataset", "Home Credit")
    else:
        rows, cols = df.shape
        rate = df['TARGET'].mean() * 100
        missing_cols = df.isnull().any().sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{rows:,}")
        c2.metric("Total Columns", f"{cols}")
        c3.metric("Default Rate", f"{rate:.2f}%")
        c4.metric("Cols with Missing", f"{missing_cols}")

    st.markdown("---")

    
    st.markdown("### ⚠️ The 5 Leakage Sins")
    cols = st.columns(5)
    for i, (sin_num, info) in enumerate(SIN_INFO.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="sin-card">
                <div style="font-size:1.8rem">{info['icon']}</div>
                <div style="font-weight:700;color:#F0D080;margin:0.3rem 0">Sin {sin_num}</div>
                <div style="font-size:0.85rem;color:#9AA5BC">{info['name']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ How To Use This Dashboard")
    st.markdown("""
    | Page | What You Learn |
    |------|---------------|
    | 📊 AUC Explainer | What AUC means and why it can lie |
    | 🔍 Sin Explorer | Deep dive into each leakage sin — story, mechanism, AUC impact |
    | 📈 Master Comparison | All 5 sins side by side — see which sin inflates AUC the most |
    | 🤖 Auto Detector | Upload any CSV — get an instant leakage risk report |
    """)



elif page == "📊 AUC Explainer":
    st.markdown("## 📊 What Is AUC — And Why Can It Lie?")

    st.markdown("""
    AUC (Area Under the ROC Curve) answers one question:

    > **If I pick one person who defaulted and one who didn't — what is the probability my model scored the defaulter higher?**

    - **0.50** → coin flip. Your model learned nothing.
    - **0.70** → decent. Learning real signal.
    - **0.90+** → suspicious. Check for leakage.
    """)

    from visualize import plot_auc_explainer
    fig = plot_auc_explainer(save=False)
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🧮 AUC By Hand — 5 Person Example")
    st.markdown("""
    | Person | Defaulted? | Model Score |
    |--------|-----------|-------------|
    | A | ✅ Yes | 0.90 |
    | B | ✅ Yes | 0.70 |
    | C | ❌ No  | 0.80 |
    | D | ❌ No  | 0.40 |
    | E | ❌ No  | 0.30 |

    **Pairs to check** (each defaulter vs each non-defaulter):
    - A vs C: 0.90 > 0.80 ✅
    - A vs D: 0.90 > 0.40 ✅
    - A vs E: 0.90 > 0.30 ✅
    - B vs C: 0.70 < 0.80 ❌  ← model got this wrong
    - B vs D: 0.70 > 0.40 ✅
    - B vs E: 0.70 > 0.30 ✅

    **AUC = 5/6 = 0.833**

    Person C scored higher than defaulter B — one mistake. Leakage hides mistakes like this by letting the model peek at the answer.
    """)



elif page == "🔍 Sin Explorer":
    st.markdown("## 🔍 Sin Explorer")
    st.markdown("Select a leakage sin to see its story, mechanism, and AUC impact.")

    sin_choice = st.selectbox(
        "Choose a Leakage Sin",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"Sin {x} — {SIN_INFO[x]['name']}"
    )

    info = SIN_INFO[sin_choice]

    
    st.markdown(f"### {info['icon']} Sin {sin_choice}: {info['name']}")

    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📖 The Story**")
        st.info(info['story'])
    with c2:
        st.markdown("**❌ The Sin**")
        st.error(info['the_sin'])
    with c3:
        st.markdown("**✅ The Fix**")
        st.success(info['the_fix'])

    st.markdown(f"**🔎 Detection:** `{info['detection']}`")
    st.markdown("---")

    
    st.markdown("### 📊 AUC Impact")

    sin_results, is_demo = get_sin_results()

    if is_demo:
        st.warning("⚠️ Demo mode — place `application_train.csv` in `data/` to see real results.")

    if sin_results:
        r = next((x for x in sin_results if x['sin'] == sin_choice), None)
        if r:
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div style="color:#9AA5BC">Honest AUC</div><div class="auc-honest">{r["honest_auc"]:.4f}</div><div style="color:#9AA5BC;font-size:0.8rem">Real production performance</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with mc2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div style="color:#9AA5BC">Leaky AUC</div><div class="auc-leaky">{r["leaky_auc"]:.4f}</div><div style="color:#9AA5BC;font-size:0.8rem">Fake inflated score</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with mc3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div style="color:#9AA5BC">Inflation</div><div class="inflation">+{r["inflation"]:.4f}</div><div style="color:#9AA5BC;font-size:0.8rem">The lie</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("")
            from visualize import plot_sin_detail
            fig = plot_sin_detail(r['sin'], r['honest_auc'], r['leaky_auc'], r['name'], save=False)
            st.pyplot(fig)



elif page == "📈 Master Comparison":
    st.markdown("## 📈 Master Comparison — All 5 Sins")
    st.markdown("Which sin inflates AUC the most? Which is the most dangerous?")

    sin_results, is_demo = get_sin_results()

    if is_demo:
        st.warning("⚠️ Demo mode — place `application_train.csv` in `data/` to see real results.")

    if sin_results:
        from visualize import plot_auc_comparison
        fig = plot_auc_comparison(sin_results, save=False)
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 📋 Results Table")

        table_data = []
        for r in sin_results:
            inf = r['inflation']
            severity = "🔴 CRITICAL" if inf>0.15 else "🟠 HIGH" if inf>0.08 else "🟡 MODERATE" if inf>0.03 else "🟢 LOW"
            table_data.append({
                'Sin': f"Sin {r['sin']}",
                'Name': r['name'],
                'Honest AUC': f"{r['honest_auc']:.4f}",
                'Leaky AUC': f"{r['leaky_auc']:.4f}",
                'Inflation': f"+{r['inflation']:.4f}",
                'Severity': severity
            })

        st.table(pd.DataFrame(table_data))

        worst = max(sin_results, key=lambda x: x['inflation'])
        st.markdown(f"### 🔴 Most Dangerous Sin")
        st.error(f"**Sin {worst['sin']}: {worst['name']}** inflated AUC by **+{worst['inflation']:.4f}** — the biggest lie in this dataset.")



elif page == "🤖 Auto Detector":
    st.markdown("## 🤖 Auto Leakage Detector")
    st.markdown("Upload any CSV dataset and get an instant leakage risk scan.")

    uploaded = st.file_uploader("Upload a CSV file", type=['csv'])

    col_left, col_right = st.columns(2)
    with col_left:
        target_col = st.text_input("Target column name", value="TARGET")
    with col_right:
        id_col = st.text_input("ID column name (optional)", value="SK_ID_CURR")

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.success(f"✅ Loaded: {df_upload.shape[0]:,} rows × {df_upload.shape[1]} columns")

            if target_col not in df_upload.columns:
                st.error(f"Column '{target_col}' not found. Available columns: {list(df_upload.columns[:10])}")
            else:
                if st.button("🔍 Run Leakage Scan", type="primary"):
                    with st.spinner("Scanning for leakage..."):
                        from leakage_detector import LeakageDetector
                        id_input = id_col if id_col in df_upload.columns else None
                        detector = LeakageDetector(df_upload, target_col=target_col, id_col=id_input)
                        report = detector.run_full_scan()

                    st.markdown("---")
                    st.markdown("### 📋 Scan Results")

                    ICONS = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢','CLEAN':'✅','UNKNOWN':'⚪'}
                    sin_keys = [k for k in report if k not in ['score','level'] and isinstance(report[k], dict)]

                    for key in sin_keys:
                        r = report[key]
                        icon = ICONS.get(r['risk'], '?')
                        with st.expander(f"{icon} {r['name']} — {r['risk']}"):
                            st.markdown(f"**Finding:** {r['message']}")
                            st.markdown(f"**Fix:** {r['fix']}")
                            if 'flagged' in r and r['flagged']:
                                st.markdown("**Flagged features:**")
                                for feat, corr in list(r['flagged'].items())[:5]:
                                    st.code(f"{feat}: correlation={corr:.4f}")

                    st.markdown("---")
                    score = report['score']
                    level = report['level']
                    icon  = ICONS.get(level,'?')

                    if level == 'CLEAN':
                        st.success(f"{icon} Overall Score: {score}/100 — {level}. Dataset passed all checks.")
                    elif level in ['LOW', 'MEDIUM']:
                        st.warning(f"{icon} Overall Score: {score}/100 — {level}. Review flagged items.")
                    else:
                        st.error(f"{icon} Overall Score: {score}/100 — {level}. Fix issues before training.")

                    from visualize import plot_detector_radar
                    fig = plot_detector_radar(report, save=False)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.markdown("""
        **How to use:**
        1. Upload any tabular CSV dataset
        2. Enter the name of your target column (what you are predicting)
        3. Enter your ID column name (optional but recommended)
        4. Click **Run Leakage Scan**
        5. Get an instant risk report across all 5 leakage archetypes

        **Works on any dataset** — not just Home Credit.
        """)
        st.info("💡 Try uploading `data/application_train.csv` to scan the Home Credit dataset itself.")
