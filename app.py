from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, confusion_matrix, recall_score,
    precision_score, f1_score, accuracy_score
)
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PokéClassifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Nunito:wght@400;600;700;800&display=swap');

  :root {
    --gold:    #FFD700;
    --red:     #E3350D;
    --blue:    #1A73E8;
    --dark:    #0D0D1A;
    --card:    #13132A;
    --border:  #2A2A4A;
    --text:    #E8E8F8;
    --muted:   #8888AA;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--dark);
    color: var(--text);
    font-family: 'Nunito', sans-serif;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0A0A18;
    border-right: 1px solid var(--border);
  }

  /* Header */
  .poke-header {
    text-align: center;
    padding: 2rem 0 1rem;
  }
  .poke-header h1 {
    font-family: 'Press Start 2P', monospace;
    font-size: 1.6rem;
    color: var(--gold);
    text-shadow: 3px 3px 0 var(--red), 6px 6px 0 #000;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
  }
  .poke-header p {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 600;
  }

  /* Cards */
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: var(--gold); }
  .stat-card .label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }
  .stat-card .value {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--gold);
  }

  /* Type badges */
  .type-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    margin: 2px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* Section titles */
  .section-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 0.75rem;
    color: var(--gold);
    letter-spacing: 1px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 1.5rem 0 1rem;
  }

  /* Best model banner */
  .best-banner {
    background: linear-gradient(135deg, #1a1a00, #2a2200);
    border: 2px solid var(--gold);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
  }
  .best-banner .model-name {
    font-family: 'Press Start 2P', monospace;
    font-size: 0.9rem;
    color: var(--gold);
  }

  /* Prediction box */
  .pred-legendary {
    background: linear-gradient(135deg, #1a0000, #2e0000);
    border: 2px solid var(--red);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }
  .pred-normal {
    background: linear-gradient(135deg, #001a0a, #002e14);
    border: 2px solid #00C878;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }
  .pred-label {
    font-family: 'Press Start 2P', monospace;
    font-size: 1.1rem;
  }

  /* Metric rows */
  .metric-row {
    display: flex; gap: 12px; margin-bottom: 12px;
  }

  /* Override streamlit button style */
  .stButton > button {
    background: var(--gold) !important;
    color: #000 !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.65rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    cursor: pointer;
    transition: transform 0.1s;
  }
  .stButton > button:hover { transform: translateY(-2px); }

  /* Tab overrides */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
  }

  /* Slider label */
  .stSlider label { color: var(--muted) !important; font-size: 0.85rem !important; }

  /* Selectbox */
  .stSelectbox label { color: var(--muted) !important; }

  /* Progress bar */
  .stProgress > div > div { background: var(--gold) !important; }
</style>
""", unsafe_allow_html=True)

# ── Type Colors ───────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "Normal": "#A8A878", "Fire": "#F08030", "Water": "#6890F0", "Electric": "#F8D030",
    "Grass": "#78C850", "Ice": "#98D8D8", "Fighting": "#C03028", "Poison": "#A040A0",
    "Ground": "#E0C068", "Flying": "#A890F0", "Psychic": "#F85888", "Bug": "#A8B820",
    "Rock": "#B8A038", "Ghost": "#705898", "Dragon": "#7038F8", "Dark": "#705848",
    "Steel": "#B8B8D0", "Fairy": "#EE99AC", "None": "#555566",
}

# ── Data Loading ──────────────────────────────────────────────────────────────


@st.cache_data
def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Remove mega evolutions, regional variants, tera forms
    if 'name' in df.columns:
        df = df[~df['name'].str.contains('-Mega', na=False)]
        tera_idx = df[df['name'].str.contains('Tera', na=False)].index
        df = df.drop(tera_idx, errors='ignore')
        df = df.drop_duplicates()

    # Fill type_2
    if 'type_2' in df.columns:
        df['type_2'] = df['type_2'].fillna('None')

    return df


# ── Model Training ────────────────────────────────────────────────────────────
FEATURE_COLS = ['type_1', 'type_2', 'hp', 'attack', 'defense',
                'sp_attack', 'sp_defense', 'speed',
                'base_stat_total', 'base_experience', 'capture_rate',
                'is_baby', 'is_mythical']
TARGET_COL = 'is_legendary'

MODEL_MAP = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbours": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}


@st.cache_resource
def train_models(df_hash):
    return None  # placeholder — actual training below


def run_training(df):
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    # Handle base_experience NaN
    if 'base_experience' in X.columns:
        X['base_experience'] = X['base_experience'].fillna(
            X['base_experience'].median())

    numerical_cols = X.select_dtypes(include='number').columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('encoder', OneHotEncoder(
        drop='first', handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numerical_cols),
        ('cat', cat_pipe, categorical_cols),
    ])

    results = {}
    for name, clf in MODEL_MAP.items():
        pipe = Pipeline([
            ('prep', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        proba = None
        if hasattr(clf, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            proba = y_proba

        results[name] = {
            "pipeline": pipe,
            "y_test": y_test,
            "y_pred": y_pred,
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    return results, available_features, numerical_cols, categorical_cols


# ── Matplotlib dark theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#13132A",
    "axes.facecolor": "#13132A",
    "axes.edgecolor": "#2A2A4A",
    "axes.labelcolor": "#8888AA",
    "xtick.color": "#8888AA",
    "ytick.color": "#8888AA",
    "text.color": "#E8E8F8",
    "grid.color": "#2A2A4A",
    "grid.linestyle": "--",
    "font.family": "sans-serif",
})

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="poke-header">
  <h1>⚡ PokéClassifier ⚡</h1>
  <p>Legendary Detection Engine · Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data Source")
    uploaded = st.file_uploader("Upload pokemon_complete.csv", type=["csv"])

    if uploaded:
        df_raw = load_and_clean_data(uploaded)
        st.success(f"✅ {len(df_raw)} Pokémon loaded")
    else:
        st.info("⬆️ Upload your CSV to begin")
        df_raw = None

    st.markdown("---")
    st.markdown("### 🔬 Training Configuration")
    test_size = st.slider("Test split %", 10, 40, 20, step=5)
    use_smote = st.toggle("Use SMOTE (oversampling)", value=True)

    st.markdown("---")
    st.markdown("### 🤖 Models")
    selected_models = st.multiselect(
        "Select models to train",
        list(MODEL_MAP.keys()),
        default=list(MODEL_MAP.keys()),
    )

    train_btn = st.button("🚀 Train Models", disabled=(
        df_raw is None or len(selected_models) == 0))

# ── Main Tabs ─────────────────────────────────────────────────────────────────
if df_raw is None:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #555566;">
      <div style="font-size:5rem">⚡</div>
      <div style="font-family:'Press Start 2P',monospace; font-size:0.8rem; color:#FFD700; margin: 1rem 0;">
        Upload your dataset to begin
      </div>
      <p>The app expects <code>pokemon_complete.csv</code> with stats like<br>
      HP, Attack, Defense, Speed, Type, Capture Rate, and the <code>is_legendary</code> target.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
tab_eda, tab_train, tab_predict = st.tabs(
    ["📊 Explore Data", "🧠 Train & Compare", "🎯 Predict Legendary"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════
with tab_eda:
    df = df_raw.copy()

    # KPI row
    total = len(df)
    legendary = int(df[TARGET_COL].sum()
                    ) if TARGET_COL in df.columns else "N/A"
    types = df['type_1'].nunique() if 'type_1' in df.columns else "N/A"
    mythical = int(df['is_mythical'].sum()
                   ) if 'is_mythical' in df.columns else "N/A"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="stat-card"><div class="label">Total Pokémon</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(
            f'<div class="stat-card"><div class="label">Legendary</div><div class="value">{legendary}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(
            f'<div class="stat-card"><div class="label">Types</div><div class="value">{types}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(
            f'<div class="stat-card"><div class="label">Mythical</div><div class="value">{mythical}</div></div>', unsafe_allow_html=True)

    # st.markdown('<div class="section-title">CLASS BALANCE</div>',
    #             unsafe_allow_html=True)
    # col_a, col_b = st.columns([1, 2])
    # with col_a:
    #     if TARGET_COL in df.columns:
    #         counts = df[TARGET_COL].value_counts()
    #         pct_leg = counts.get(1, 0) / total * 100
    #         pct_norm = counts.get(0, 0) / total * 100
    #         st.markdown(f"""
    #         <div style="background:#13132A;border:1px solid #2A2A4A;border-radius:12px;padding:1.2rem;margin-top:0.5rem">
    #           <div style="margin-bottom:10px">
    #             <span style="color:#E3350D;font-weight:800">LEGENDARY</span>
    #             <span style="float:right;color:#FFD700;font-weight:800">{counts.get(1, 0)} ({pct_leg:.1f}%)</span>
    #           </div>
    #           <div>
    #             <span style="color:#00C878;font-weight:800">NORMAL</span>
    #             <span style="float:right;color:#FFD700;font-weight:800">{counts.get(0, 0)} ({pct_norm:.1f}%)</span>
    #           </div>
    #           <div style="margin-top:12px;background:#2A2A4A;border-radius:6px;height:10px">
    #             <div style="background:#E3350D;width:{pct_leg:.1f}%;height:100%;border-radius:6px"></div>
    #           </div>
    #           <div style="font-size:0.75rem;color:#555566;margin-top:6px">⚠️ Class imbalance → SMOTE recommended</div>
    #         </div>
    #         """, unsafe_allow_html=True)

    # with col_b:
    #     if TARGET_COL in df.columns:
    #         fig, ax = plt.subplots(figsize=(6, 3))
    #         bars = ax.bar(["Normal (0)", "Legendary (1)"],
    #                       [counts.get(0, 0), counts.get(1, 0)],
    #                       color=["#00C878", "#E3350D"], width=0.5, edgecolor='none')
    #         for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
    #             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
    #                     str(val), ha='center', fontsize=11, fontweight='bold', color='#FFD700')
    #         ax.set_ylabel("Count")
    #         ax.set_title("Legendary vs Normal Distribution")
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         st.pyplot(fig, use_container_width=True)
    #         plt.close()

    # Type distribution
    st.markdown('<div class="section-title">TYPE DISTRIBUTION</div>',
                unsafe_allow_html=True)
    if 'type_1' in df.columns:
        type_counts = df['type_1'].value_counts()
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = [TYPE_COLORS.get(t, "#555566") for t in type_counts.index]
        bars = ax.bar(type_counts.index, type_counts.values,
                      color=colors, edgecolor='none')
        ax.set_title("Pokémon by Primary Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=35, ha='right', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Stats comparison legendary vs normal
    st.markdown('<div class="section-title">STAT COMPARISON · LEGENDARY vs NORMAL</div>',
                unsafe_allow_html=True)
    stat_cols = [c for c in ['hp', 'attack', 'defense', 'sp_attack',
                             'sp_defense', 'speed', 'base_stat_total'] if c in df.columns]
    if stat_cols and TARGET_COL in df.columns:
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()
        for i, col in enumerate(stat_cols):
            ax = axes[i]
            data_leg = df[df[TARGET_COL] == 1][col].dropna()
            data_norm = df[df[TARGET_COL] == 0][col].dropna()
            ax.hist(data_norm, bins=25, alpha=0.7, color='#00C878',
                    label='Normal', edgecolor='none')
            ax.hist(data_leg, bins=25, alpha=0.7, color='#E3350D',
                    label='Legendary', edgecolor='none')
            ax.set_title(col.replace('_', ' ').title(), fontsize=9)
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for j in range(len(stat_cols), len(axes)):
            axes[j].set_visible(False)

        handles = [mpatches.Patch(color='#00C878', label='Normal'),
                   mpatches.Patch(color='#E3350D', label='Legendary')]
        fig.legend(handles=handles, loc='lower right',
                   fontsize=9, framealpha=0.3)
        fig.suptitle("Stat Distributions", fontsize=12,
                     color='#FFD700', y=1.01)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Correlation heatmap
    st.markdown('<div class="section-title">CORRELATION HEATMAP</div>',
                unsafe_allow_html=True)
    num_df = df.select_dtypes(include='number')
    if len(num_df.columns) > 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
        sns.heatmap(num_df.corr(), ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
                    linecolor='#0D0D1A', linewidths=0.5, mask=mask,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlations", pad=12)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — TRAIN & COMPARE
# ══════════════════════════════════════════════════════════════
with tab_train:
    if "results" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#555566">
          <div style="font-size:3.5rem">🧠</div>
          <div style="font-family:'Press Start 2P',monospace;color:#FFD700;font-size:0.75rem;margin:1rem">
            Click "Train Models" in the sidebar to start
          </div>
        </div>
        """, unsafe_allow_html=True)

    if train_btn:
        MODEL_MAP_SELECTED = {k: v for k,
                              v in MODEL_MAP.items() if k in selected_models}
        if not MODEL_MAP_SELECTED:
            st.warning("Select at least one model.")
        else:
            progress = st.progress(0, text="Preparing data…")
            df_clean = df_raw.copy()
            avail = [c for c in FEATURE_COLS if c in df_clean.columns]
            X = df_clean[avail].copy()
            y = df_clean[TARGET_COL].copy()
            if 'base_experience' in X.columns:
                X['base_experience'] = X['base_experience'].fillna(
                    X['base_experience'].median())

            num_cols = X.select_dtypes(include='number').columns.tolist()
            cat_cols = X.select_dtypes(include='object').columns.tolist()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y)

            num_pipe = Pipeline([('scaler', StandardScaler())])
            cat_pipe = Pipeline([('encoder', OneHotEncoder(
                drop='first', handle_unknown='ignore', sparse_output=False))])
            preprocessor = ColumnTransformer(
                [('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

            results = {}
            for i, (name, clf) in enumerate(MODEL_MAP_SELECTED.items()):
                progress.progress((i) / len(MODEL_MAP_SELECTED),
                                  text=f"Training {name}…")
                steps = [('prep', preprocessor)]
                if use_smote:
                    steps.append(('smote', SMOTE(random_state=42)))
                steps.append(('clf', clf))
                pipe = Pipeline(steps)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                results[name] = {
                    "pipeline": pipe,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "recall": recall_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "cm": confusion_matrix(y_test, y_pred),
                }
            progress.progress(1.0, text="Done! ✅")

            st.session_state["results"] = results
            st.session_state["feature_cols"] = avail
            st.session_state["num_cols"] = num_cols
            st.session_state["cat_cols"] = cat_cols

    if "results" in st.session_state:
        results = st.session_state["results"]

        # Best model
        best_name = max(results, key=lambda k: results[k]["recall"])
        best = results[best_name]
        st.markdown(f"""
        <div class="best-banner">
          <div style="color:#888;font-size:0.8rem;margin-bottom:6px">🏆 BEST MODEL BY RECALL</div>
          <div class="model-name">{best_name}</div>
          <div style="color:#FFD700;font-size:1.3rem;font-weight:800;margin-top:6px">
            {best["recall"]*100:.1f}% Recall
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Recall comparison bar chart
        st.markdown(
            '<div class="section-title">RECALL COMPARISON</div>', unsafe_allow_html=True)
        names = list(results.keys())
        recalls = [results[n]["recall"]*100 for n in names]
        fig, ax = plt.subplots(figsize=(10, 4))
        bar_colors = ["#FFD700" if n ==
                      best_name else "#2A2A6A" for n in names]
        bars = ax.barh(names, recalls, color=bar_colors,
                       edgecolor='none', height=0.5)
        for bar, val in zip(bars, recalls):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va='center', fontsize=10, fontweight='bold',
                    color='#FFD700')
        ax.set_xlim(0, 115)
        ax.set_xlabel("Recall (%)")
        ax.set_title("Model Recall — Legendary Detection")
        ax.axvline(x=80, color='#E3350D', ls='--', alpha=0.5, lw=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # All metrics table
        st.markdown('<div class="section-title">ALL METRICS</div>',
                    unsafe_allow_html=True)
        metrics_data = []
        for n, r in results.items():
            metrics_data.append({
                "Model": n,
                "Recall": f"{r['recall']*100:.1f}%",
                "Precision": f"{r['precision']*100:.1f}%",
                "F1 Score": f"{r['f1']*100:.1f}%",
                "Accuracy": f"{r['accuracy']*100:.1f}%",
            })
        metrics_df = pd.DataFrame(metrics_data).set_index("Model")
        st.dataframe(metrics_df, use_container_width=True)

        # Confusion matrices
        st.markdown(
            '<div class="section-title">CONFUSION MATRICES</div>', unsafe_allow_html=True)
        n_models = len(results)
        cols = st.columns(min(n_models, 3))
        for i, (name, res) in enumerate(results.items()):
            with cols[i % 3]:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                cm = res["cm"]
                sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap='YlOrRd',
                            linewidths=0.5, linecolor='#0D0D1A',
                            xticklabels=['Normal', 'Legendary'],
                            yticklabels=['Normal', 'Legendary'])
                ax.set_title(name, fontsize=9, color='#FFD700')
                ax.set_xlabel("Predicted", fontsize=8)
                ax.set_ylabel("Actual", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab_predict:
    if "results" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#555566">
          <div style="font-size:3.5rem">🎯</div>
          <div style="font-family:'Press Start 2P',monospace;color:#FFD700;font-size:0.75rem;margin:1rem">
            Train the models first to enable predictions
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    results = st.session_state["results"]
    feature_cols = st.session_state["feature_cols"]
    best_model_name = max(results, key=lambda k: results[k]["recall"])

    st.markdown("### 🔮 Enter a Pokémon's Stats")
    all_types = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
                 "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]

    c1, c2 = st.columns(2)
    with c1:
        t1 = st.selectbox("Primary Type", all_types, index=0)
        hp = st.slider("HP", 1, 255, 80)
        attack = st.slider("Attack", 1, 255, 80)
        defense = st.slider("Defense", 1, 255, 70)
        sp_attack = st.slider("Sp. Attack", 1, 255, 80)
    with c2:
        t2 = st.selectbox("Secondary Type", ["None"] + all_types, index=0)
        sp_defense = st.slider("Sp. Defense", 1, 255, 70)
        speed = st.slider("Speed", 1, 255, 75)
        capture_rate = st.slider("Capture Rate", 3, 255, 45)
        base_exp = st.slider("Base Experience", 36, 608, 200)

    bst = hp + attack + defense + sp_attack + sp_defense + speed
    st.markdown(f"**Base Stat Total (auto-calculated): `{bst}`**")

    is_baby = st.checkbox("Is Baby Pokémon?", value=False)
    is_mythical = st.checkbox("Is Mythical?", value=False)

    chosen_model = st.selectbox("Model to use for prediction",
                                list(results.keys()),
                                index=list(results.keys()).index(best_model_name))

    if st.button("⚡ PREDICT LEGENDARY STATUS"):
        input_data = {}
        for col in feature_cols:
            if col == 'type_1':
                input_data[col] = t1
            elif col == 'type_2':
                input_data[col] = t2
            elif col == 'hp':
                input_data[col] = hp
            elif col == 'attack':
                input_data[col] = attack
            elif col == 'defense':
                input_data[col] = defense
            elif col == 'sp_attack':
                input_data[col] = sp_attack
            elif col == 'sp_defense':
                input_data[col] = sp_defense
            elif col == 'speed':
                input_data[col] = speed
            elif col == 'base_stat_total':
                input_data[col] = bst
            elif col == 'base_experience':
                input_data[col] = base_exp
            elif col == 'capture_rate':
                input_data[col] = capture_rate
            elif col == 'is_baby':
                input_data[col] = int(is_baby)
            elif col == 'is_mythical':
                input_data[col] = int(is_mythical)

        input_df = pd.DataFrame([input_data])
        pipe = results[chosen_model]["pipeline"]
        pred = pipe.predict(input_df)[0]

        try:
            proba = pipe.predict_proba(input_df)[0][1]
            proba_str = f"{proba*100:.1f}%"
        except:
            proba_str = "N/A"

        if pred == 1:
            st.markdown(f"""
            <div class="pred-legendary">
              <div style="font-size:3rem;margin-bottom:0.5rem">🌟</div>
              <div class="pred-label" style="color:#E3350D">LEGENDARY</div>
              <div style="color:#FFD700;font-size:1.4rem;font-weight:800;margin-top:8px">
                Confidence: {proba_str}
              </div>
              <div style="color:#888;font-size:0.8rem;margin-top:4px">via {chosen_model}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-normal">
              <div style="font-size:3rem;margin-bottom:0.5rem">🟢</div>
              <div class="pred-label" style="color:#00C878">NORMAL</div>
              <div style="color:#FFD700;font-size:1.4rem;font-weight:800;margin-top:8px">
                Legendary Probability: {proba_str}
              </div>
              <div style="color:#888;font-size:0.8rem;margin-top:4px">via {chosen_model}</div>
            </div>
            """, unsafe_allow_html=True)

        # Type badge display
        badge_t1 = TYPE_COLORS.get(t1, "#555566")
        badge_t2 = TYPE_COLORS.get(t2, "#555566")
        st.markdown(f"""
        <div style="margin-top:1rem">
          <span class="type-badge" style="background:{badge_t1};color:#fff">{t1}</span>
          {"" if t2 == "None" else f'<span class="type-badge" style="background:{badge_t2};color:#fff">{t2}</span>'}
          &nbsp;|&nbsp; BST: <b>{bst}</b>
          &nbsp;|&nbsp; Capture Rate: <b>{capture_rate}</b>
        </div>
        """, unsafe_allow_html=True)
