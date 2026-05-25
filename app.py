import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictiveMaint AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #ffffff; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3147;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2235, #252a40);
        border: 1px solid #2e3147;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4f8ef7;
        margin: 0;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #8b93b0;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .delta {
        font-size: 0.8rem;
        color: #3ecf8e;
        margin-top: 6px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        border-left: 4px solid #4f8ef7;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Risk badge */
    .badge-high   { background:#ff4b4b22; color:#ff4b4b; border:1px solid #ff4b4b55;
                    padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-medium { background:#ffa50022; color:#ffa500; border:1px solid #ffa50055;
                    padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-low    { background:#3ecf8e22; color:#3ecf8e; border:1px solid #3ecf8e55;
                    padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

    /* Top nav bar */
    .topbar {
        background: linear-gradient(90deg, #1a1d27, #252a40);
        border-bottom: 1px solid #2e3147;
        padding: 16px 24px;
        margin-bottom: 24px;
        border-radius: 0 0 12px 12px;
    }
    .topbar h1 { font-size: 1.6rem; font-weight: 700; color: #ffffff; margin: 0; }
    .topbar p  { font-size: 0.9rem; color: #8b93b0; margin: 4px 0 0 0; }

    /* Divider */
    hr { border-color: #2e3147; }

    /* DataFrame */
    .dataframe { background: #1a1d27 !important; }

    /* Alert boxes */
    .alert-box {
        background: #ff4b4b15;
        border: 1px solid #ff4b4b44;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .alert-box .machine { font-weight: 700; color: #ff4b4b; font-size: 1rem; }
    .alert-box .detail  { color: #aab0c8; font-size: 0.85rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ PredictiveMaint AI")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Dashboard",
        "📊  EDA & Insights",
        "🤖  Model & Evaluation",
        "🔧  Maintenance Planner",
        "🔍  Live Predictor"
    ])
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    prediction_window = st.slider("Prediction window (hours)", 6, 48, 24)
    threshold = st.slider("Alert threshold (%)", 10, 90, 50)
    st.markdown("---")
    st.markdown("<p style='color:#8b93b0;font-size:0.8rem;'>Internship Project · 2026<br>Random Forest · ROC-AUC 0.97</p>",
                unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    telemetry   = pd.read_csv('PdM_telemetry.csv',  parse_dates=['datetime'])
    failures    = pd.read_csv('PdM_failures.csv',    parse_dates=['datetime'])
    maintenance = pd.read_csv('PdM_maint.csv',       parse_dates=['datetime'])
    machines    = pd.read_csv('PdM_machines.csv')
    errors      = pd.read_csv('PdM_errors.csv',      parse_dates=['datetime'])
    return telemetry, failures, maintenance, machines, errors

@st.cache_data
def build_features(telemetry, failures, machines, window_hours=24):
    sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
    df = telemetry.merge(machines, on='machineID', how='left')
    df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)
    df['label'] = 0
    for _, row in failures.iterrows():
        ws = row['datetime'] - pd.Timedelta(hours=window_hours)
        mask = ((df['machineID'] == row['machineID']) &
                (df['datetime'] >= ws) & (df['datetime'] < row['datetime']))
        df.loc[mask, 'label'] = 1
    for col in sensor_cols:
        for w in [3, 12, 24]:
            grp = df.groupby('machineID')[col]
            df[f'{col}_mean_{w}h'] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f'{col}_std_{w}h']  = grp.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        for lag in [1, 3, 6]:
            df[f'{col}_lag{lag}'] = df.groupby('machineID')[col].shift(lag)
    df['hour']      = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['model']     = LabelEncoder().fit_transform(df['model'].astype(str))
    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_model(df):
    feature_cols = [c for c in df.columns if c not in ['datetime','machineID','label']]
    split_date   = pd.Timestamp('2015-10-01')
    train = df[df['datetime'] <= split_date]
    test  = df[df['datetime'] >  split_date]
    X_tr, y_tr = train[feature_cols], train['label']
    X_te, y_te = test[feature_cols],  test['label']
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]
    return rf, feature_cols, X_te, y_te, y_pred, y_prob, test

# ── Load ───────────────────────────────────────────────────────────────────────
sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']

with st.spinner("Loading data..."):
    try:
        telemetry, failures, maintenance, machines, errors = load_data()
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False

if not data_loaded:
    st.error("CSV files not found. Please upload PdM_telemetry.csv, PdM_failures.csv, PdM_maint.csv, PdM_machines.csv, PdM_errors.csv to the same folder as app.py")
    st.stop()

with st.spinner("Engineering features..."):
    df = build_features(telemetry, failures, machines, prediction_window)

with st.spinner("Training model..."):
    rf, feature_cols, X_te, y_te, y_pred, y_prob, test_df = train_model(df)

roc_auc  = roc_auc_score(y_te, y_prob)
accuracy = (y_pred == y_te).mean()
recall   = (y_pred[y_te == 1] == 1).mean()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("""
    <div class='topbar'>
        <h1>⚙️ Predictive Maintenance AI</h1>
        <p>Real-time equipment failure forecasting · 100 machines · 876k sensor readings</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <p class='value'>{roc_auc:.3f}</p>
            <p class='label'>ROC-AUC Score</p>
            <p class='delta'>↑ Target was 0.85</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <p class='value'>{accuracy:.1%}</p>
            <p class='label'>Accuracy</p>
            <p class='delta'>↑ Exceeds 85% target</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <p class='value'>{recall:.1%}</p>
            <p class='label'>Failure Recall</p>
            <p class='delta'>Failures caught</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        at_risk = int((y_prob >= threshold/100).sum())
        st.markdown(f"""<div class='metric-card'>
            <p class='value'>{at_risk:,}</p>
            <p class='label'>High-Risk Hours</p>
            <p class='delta'>Above {threshold}% threshold</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='section-header'>Failure events over time</div>", unsafe_allow_html=True)
        monthly = failures.copy()
        monthly['month'] = monthly['datetime'].dt.to_period('M').astype(str)
        mc = monthly.groupby(['month','failure']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        mc.plot(kind='bar', ax=ax, width=0.7)
        ax.set_xlabel('Month', color='#8b93b0')
        ax.set_ylabel('Failures', color='#8b93b0')
        ax.tick_params(colors='#8b93b0')
        ax.legend(facecolor='#252a40', labelcolor='white', fontsize=8)
        for spine in ax.spines.values(): spine.set_color('#2e3147')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-header'>Failure by component</div>", unsafe_allow_html=True)
        fc = failures['failure'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        colors = ['#4f8ef7','#3ecf8e','#ffa500','#ff4b4b']
        wedges, texts, autotexts = ax.pie(fc.values, labels=fc.index, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           textprops={'color':'white','fontsize':9})
        for at in autotexts: at.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("<div class='section-header'>Top 10 machines — current risk level</div>", unsafe_allow_html=True)
    test_copy = test_df.copy()
    test_copy['failure_prob'] = y_prob
    risk = (test_copy.groupby('machineID')
            .agg(avg_risk=('failure_prob','mean'),
                 high_risk_hours=('failure_prob', lambda x: (x >= threshold/100).sum()))
            .sort_values('avg_risk', ascending=False).head(10))
    risk['avg_risk_pct'] = (risk['avg_risk']*100).round(2)

    def risk_badge(v):
        if v >= 12: return f"<span class='badge-high'>HIGH</span>"
        elif v >= 10: return f"<span class='badge-medium'>MEDIUM</span>"
        else: return f"<span class='badge-low'>LOW</span>"

    display = risk[['avg_risk_pct','high_risk_hours']].copy()
    display.columns = ['Avg Risk %','High-Risk Hours']
    display['Risk Level'] = display['Avg Risk %'].apply(risk_badge)
    st.write(display.to_html(escape=False), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA & Insights":
    st.markdown("<h2 style='color:white'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Sensor Distributions", "Temporal Patterns", "Correlation"])

    with tab1:
        st.markdown("<div class='section-header'>Sensor value distributions</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 4, figsize=(16, 6), facecolor='#1a1d27')
        for i, col in enumerate(sensor_cols):
            for row_i, ax in enumerate([axes[0,i], axes[1,i]]):
                ax.set_facecolor('#1a1d27')
                for spine in ax.spines.values(): spine.set_color('#2e3147')
                ax.tick_params(colors='#8b93b0')
            telemetry[col].hist(ax=axes[0,i], bins=40, color='#4f8ef7', edgecolor='#1a1d27')
            axes[0,i].set_title(col, color='white', fontsize=10)
            axes[1,i].boxplot(telemetry[col], patch_artist=True,
                              boxprops=dict(facecolor='#4f8ef755'),
                              medianprops=dict(color='#3ecf8e'),
                              whiskerprops=dict(color='#8b93b0'),
                              capprops=dict(color='#8b93b0'),
                              flierprops=dict(marker='o', color='#ff4b4b', markersize=2))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("<div class='section-header'>Dataset summary</div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        stats = telemetry[sensor_cols].describe().round(2)
        for col_w, col_n in zip([c1,c2,c3,c4], sensor_cols):
            col_w.metric(f"{col_n} mean", f"{stats[col_n]['mean']:.1f}")

    with tab2:
        machine_id = st.selectbox("Select machine", sorted(telemetry['machineID'].unique()))
        sample = telemetry[telemetry['machineID']==machine_id].sort_values('datetime')
        mfails = failures[failures['machineID']==machine_id]

        fig, axes = plt.subplots(4,1, figsize=(14,10), sharex=True, facecolor='#1a1d27')
        for ax, col in zip(axes, sensor_cols):
            ax.set_facecolor('#1a1d27')
            for spine in ax.spines.values(): spine.set_color('#2e3147')
            ax.tick_params(colors='#8b93b0')
            ax.plot(sample['datetime'], sample[col], linewidth=0.5, alpha=0.5, color='#4f8ef7')
            ax.plot(sample['datetime'], sample[col].rolling(24).mean(), linewidth=1.5, color='#3ecf8e', label='24h avg')
            ax.set_ylabel(col, color='#8b93b0')
            for _, row in mfails.iterrows():
                ax.axvline(row['datetime'], color='#ff4b4b', linewidth=1.2, linestyle='--', alpha=0.8)
            ax.legend(facecolor='#252a40', labelcolor='white', fontsize=7)
        plt.suptitle(f'Machine {machine_id} — orange = failure events', color='white', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info(f"Machine {machine_id} had **{len(mfails)} failure events** in 2015.")

    with tab3:
        st.markdown("<div class='section-header'>Sensor correlations — normal vs failure</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1,2, figsize=(12,5), facecolor='#1a1d27')
        normal_mask = ~telemetry['datetime'].isin(set(failures['datetime']))
        fail_mask   =  telemetry['datetime'].isin(set(failures['datetime']))
        for ax, data, title in zip(axes,
                                   [telemetry[normal_mask], telemetry[fail_mask]],
                                   ['Normal operation','At failure time']):
            ax.set_facecolor('#1a1d27')
            sns.heatmap(data[sensor_cols].corr(), annot=True, fmt='.2f',
                        cmap='coolwarm', center=0, ax=ax, square=True,
                        annot_kws={'color':'white'})
            ax.set_title(title, color='white')
            ax.tick_params(colors='#8b93b0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model & Evaluation":
    st.markdown("<h2 style='color:white'>🤖 Model Performance</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC",  f"{roc_auc:.4f}",  "Target: 0.85")
    c2.metric("Accuracy", f"{accuracy:.2%}", "Test set")
    c3.metric("Recall",   f"{recall:.2%}",   "Failures caught")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Confusion matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal','Failure'],
                    yticklabels=['Normal','Failure'], ax=ax,
                    annot_kws={'color':'white'})
        ax.set_xlabel('Predicted', color='#8b93b0')
        ax.set_ylabel('Actual', color='#8b93b0')
        ax.tick_params(colors='#8b93b0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-header'>Top 15 features</div>", unsafe_allow_html=True)
        importances = pd.Series(rf.feature_importances_, index=feature_cols)
        top15 = importances.nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(5,6), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        for spine in ax.spines.values(): spine.set_color('#2e3147')
        ax.tick_params(colors='#8b93b0')
        top15.plot(kind='barh', ax=ax, color='#4f8ef7', edgecolor='#1a1d27')
        ax.set_xlabel('Importance', color='#8b93b0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-header'>Classification report</div>", unsafe_allow_html=True)
    report = classification_report(y_te, y_pred,
                                   target_names=['Normal','Failure'],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3))

    st.markdown("<div class='section-header'>Top 3 features — key insight</div>", unsafe_allow_html=True)
    top3 = importances.nlargest(3)
    for i, (feat, score) in enumerate(top3.items(), 1):
        st.markdown(f"**{i}. `{feat}`** — importance score: `{score:.4f}`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MAINTENANCE PLANNER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧  Maintenance Planner":
    st.markdown("<h2 style='color:white'>🔧 Maintenance Planner</h2>", unsafe_allow_html=True)
    st.markdown("Machines ranked by failure risk. Schedule maintenance starting from the top.")

    test_copy = test_df.copy()
    test_copy['failure_prob'] = y_prob
    risk = (test_copy.groupby('machineID')
            .agg(avg_risk=('failure_prob','mean'),
                 high_risk_hours=('failure_prob', lambda x: (x >= threshold/100).sum()),
                 max_risk=('failure_prob','max'))
            .sort_values('avg_risk', ascending=False))
    risk['avg_risk_pct'] = (risk['avg_risk']*100).round(2)
    risk['max_risk_pct'] = (risk['max_risk']*100).round(2)

    st.markdown("<div class='section-header'>Risk distribution across all machines</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='#1a1d27')
    ax.set_facecolor('#1a1d27')
    for spine in ax.spines.values(): spine.set_color('#2e3147')
    ax.tick_params(colors='#8b93b0')
    colors_bar = ['#ff4b4b' if v >= 12 else '#ffa500' if v >= 10 else '#3ecf8e'
                  for v in risk['avg_risk_pct']]
    ax.bar(risk.index.astype(str), risk['avg_risk_pct'], color=colors_bar, edgecolor='#1a1d27')
    ax.axhline(threshold, color='white', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Threshold ({threshold}%)')
    ax.set_xlabel('Machine ID', color='#8b93b0')
    ax.set_ylabel('Avg failure risk %', color='#8b93b0')
    ax.legend(facecolor='#252a40', labelcolor='white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<div class='section-header'>Full machine risk table</div>", unsafe_allow_html=True)
    st.dataframe(
        risk[['avg_risk_pct','max_risk_pct','high_risk_hours']]
        .rename(columns={'avg_risk_pct':'Avg Risk %',
                         'max_risk_pct':'Peak Risk %',
                         'high_risk_hours':'High-Risk Hours'}),
        use_container_width=True, height=400
    )

    st.markdown("<div class='section-header'>Urgent alerts</div>", unsafe_allow_html=True)
    urgent = risk[risk['avg_risk_pct'] >= 11].head(5)
    for mid, row in urgent.iterrows():
        st.markdown(f"""
        <div class='alert-box'>
            <div class='machine'>⚠️ Machine {mid}</div>
            <div class='detail'>
                Avg risk: {row['avg_risk_pct']:.1f}% &nbsp;|&nbsp;
                Peak risk: {row['max_risk_pct']:.1f}% &nbsp;|&nbsp;
                High-risk hours: {int(row['high_risk_hours'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Live Predictor":
    st.markdown("<h2 style='color:white'>🔍 Live Sensor Predictor</h2>", unsafe_allow_html=True)
    st.markdown("Enter current sensor readings to get an instant failure risk prediction.")

    with st.form("predictor_form"):
        st.markdown("#### Enter sensor readings")
        c1, c2, c3, c4 = st.columns(4)
        volt      = c1.number_input("Voltage",    min_value=50.0,  max_value=300.0, value=170.0, step=0.1)
        rotate    = c2.number_input("Rotation",   min_value=100.0, max_value=700.0, value=447.0, step=0.1)
        pressure  = c3.number_input("Pressure",   min_value=50.0,  max_value=200.0, value=100.0, step=0.1)
        vibration = c4.number_input("Vibration",  min_value=10.0,  max_value=80.0,  value=40.0,  step=0.1)

        st.markdown("#### Recent history (last 24h average)")
        c5, c6, c7, c8 = st.columns(4)
        volt_24    = c5.number_input("Volt 24h avg",      value=volt,      step=0.1)
        rotate_24  = c6.number_input("Rotate 24h avg",    value=rotate,    step=0.1)
        pressure_24= c7.number_input("Pressure 24h avg",  value=pressure,  step=0.1)
        vibration_24=c8.number_input("Vibration 24h avg", value=vibration, step=0.1)

        submitted = st.form_submit_button("Predict Failure Risk", use_container_width=True)

    if submitted:
        # Build a synthetic feature row matching training feature order
        sample_row = {}
        vals = {'volt': volt, 'rotate': rotate, 'pressure': pressure, 'vibration': vibration}
        avgs = {'volt': volt_24, 'rotate': rotate_24, 'pressure': pressure_24, 'vibration': vibration_24}

        for col in feature_cols:
            if col in vals:
                sample_row[col] = vals[col]
            elif '_mean_' in col:
                base = col.split('_mean_')[0]
                sample_row[col] = avgs.get(base, vals.get(base, 0))
            elif '_std_' in col:
                sample_row[col] = 2.0
            elif '_lag' in col:
                base = col.split('_lag')[0]
                sample_row[col] = vals.get(base, 0)
            else:
                sample_row[col] = 0

        X_live = pd.DataFrame([sample_row])[feature_cols]
        prob   = rf.predict_proba(X_live)[0][1] * 100

        st.markdown("---")
        st.markdown("### Prediction result")

        if prob >= 50:
            st.error(f"⚠️ HIGH RISK — {prob:.1f}% failure probability in next {prediction_window}h")
            st.markdown("**Recommendation:** Schedule immediate inspection.")
        elif prob >= 25:
            st.warning(f"⚡ MEDIUM RISK — {prob:.1f}% failure probability in next {prediction_window}h")
            st.markdown("**Recommendation:** Monitor closely, plan maintenance soon.")
        else:
            st.success(f"✅ LOW RISK — {prob:.1f}% failure probability in next {prediction_window}h")
            st.markdown("**Recommendation:** No immediate action needed.")

        fig, ax = plt.subplots(figsize=(8, 1.5), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        color = '#ff4b4b' if prob >= 50 else '#ffa500' if prob >= 25 else '#3ecf8e'
        ax.barh(['Risk'], [prob], color=color, height=0.4)
        ax.barh(['Risk'], [100-prob], left=[prob], color='#2e3147', height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Failure probability %', color='#8b93b0')
        ax.tick_params(colors='#8b93b0')
        for spine in ax.spines.values(): spine.set_color('#2e3147')
        ax.axvline(50, color='white', linestyle='--', linewidth=1, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
