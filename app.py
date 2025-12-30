import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import os
import matplotlib.pyplot as plt

SEQLEN = 20
TARGET_COL = "Production Quality Score"

# ---------- Chargement des données et du modèle ----------

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_lstm_model():
    model_path = os.path.join("models", "modelDL.h5")
    return load_model(model_path, compile=False)

# CSV principal
DATA_PATH = "ManufacturingDataset.csv"
df = load_data(DATA_PATH)
model = load_lstm_model()

# noms de colonnes attendus
FEATURES = [
    "Temperature (°C)",
    "Machine Speed (RPM)",
    "Vibration Level (mm/s)",
    "Energy Consumption (kWh)",
]

missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes dans le CSV : {missing}")
    st.stop()

# ---------- Titre & upload CSV ----------

st.title("Dashboard LSTM – Qualité de production")

uploaded = st.sidebar.file_uploader("Charger un autre CSV (optionnel)", type=["csv"])
if uploaded is not None:
    df = load_data(uploaded)

# ---------- Sidebar : mode, historique, sliders ----------

st.sidebar.header("Mode de prédiction")
mode = st.sidebar.selectbox(
    "Choisir le mode",
    ["Valeur unique", "Prévision séquentielle (multi-pas)"]
)

n_steps = st.sidebar.slider("Nombre de pas pour la prévision séquentielle", 1, 20, 10)

hist_len = st.sidebar.slider(
    "Longueur historique affichée (points)",
    20, 200, 50, 10
)

st.sidebar.header("Paramètres machine (instant présent)")
user_vals = {}
for col in FEATURES:
    vmin = float(df[col].min())
    vmax = float(df[col].max())
    vmean = float(df[col].mean())
    user_vals[col] = st.sidebar.slider(col, vmin, vmax, vmean)

# ---------- Préparation des scalers et séquence ----------

scalerX = MinMaxScaler()
X_all = df[FEATURES].values
X_scaled = scalerX.fit_transform(X_all)

if len(X_scaled) < SEQLEN:
    st.error(f"Dataset trop court pour seqlen={SEQLEN}.")
    st.stop()

last_seq = X_scaled[-(SEQLEN - 1):]
user_point = scalerX.transform(
    np.array([[user_vals[c] for c in FEATURES]])
)[0]
seq = np.vstack([last_seq, user_point]).reshape(1, SEQLEN, len(FEATURES))

scaler_y = MinMaxScaler()
y_all = df[[TARGET_COL]].values
scaler_y.fit(y_all)

# ---------- Tabs ----------

tab_pred, tab_series, tab_corr = st.tabs(
    ["🔮 Prédiction LSTM", "📈 Séries temporelles", "📊 Corrélations / Explications"]
)

# ---------- Onglet Prédiction ----------

with tab_pred:
    st.subheader("Prédiction du Production Quality Score")

    if st.button("Lancer la prédiction"):
        if mode == "Valeur unique":
            # une seule prédiction avec la séquence construite
            y_pred_scaled = model.predict(seq)[0][0]
            y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
            preds_real = [y_pred]
        else:
            # prévision séquentielle auto-récursive sur n_steps
            seq_multi = seq.copy()
            preds_scaled = []
            for _ in range(n_steps):
                y_scaled = model.predict(seq_multi)[0][0]
                preds_scaled.append(y_scaled)
                # on duplique le dernier pas et on remplace seulement la cible implicite
                # ici on suppose que la cible est corrélée aux features, on ne modifie pas X,
                # c'est une démo simple
            preds_real = scaler_y.inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)
            ).flatten()
            y_pred = preds_real[-1]

        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score qualité prédit", f"{y_pred:.2f}")
        with col2:
            st.metric("Moyenne historique", f"{df[TARGET_COL].mean():.2f}")
        with col3:
            st.metric("Écart-type historique", f"{df[TARGET_COL].std():.2f}")

        st.write("Valeurs utilisées pour la prédiction :")
        st.json(user_vals)

        # Baseline simple (persistance) sur les 100 derniers points
        if len(df) > 101:
            test_true = df[TARGET_COL].iloc[-101:].values
            baseline_last = np.roll(test_true, 1)[1:]
            mse_base = mean_squared_error(test_true[1:], baseline_last)
            rmse_base = np.sqrt(mse_base)
            st.info(f"RMSE baseline (persistance) sur 100 pts : {rmse_base:.3f}")

        # Courbe historique vs prédiction(s)
        hist_true = df[TARGET_COL].iloc[-hist_len:].values

        plt.figure(figsize=(8, 3))
        plt.plot(range(len(hist_true)), hist_true, label="Historique")
        if mode == "Valeur unique":
            plt.scatter(len(hist_true), y_pred, color="red", label="Prédiction")
        else:
            # affiche la séquence de prévisions derrière l'historique
            x_start = len(hist_true)
            xs = list(range(x_start, x_start + len(preds_real)))
            plt.plot(xs, preds_real, color="red", marker="o", label="Prévision séquentielle")
        plt.xlabel("Temps (indices)")
        plt.ylabel("Production Quality Score")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt.gcf())

# ---------- Onglet Séries temporelles ----------

with tab_series:
    st.subheader("Séries temporelles des variables")

    ts = df.copy()
    if "Timestamp" in ts.columns:
        ts["Timestamp"] = pd.to_datetime(ts["Timestamp"], errors="coerce")
        ts = ts.set_index("Timestamp")

    cols_to_plot = FEATURES + [TARGET_COL]
    st.line_chart(ts[cols_to_plot].tail(500))

# ---------- Onglet Corrélations / Explications ----------

with tab_corr:
    st.subheader("Matrice de corrélation")
    corr = df[FEATURES + [TARGET_COL]].corr()
    st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

    st.subheader("Scatterplot qualité vs température")
    st.scatter_chart(
        df[["Temperature (°C)", TARGET_COL]].rename(
            columns={"Temperature (°C)": "Temperature", TARGET_COL: "Quality"}
        )
    )

    st.subheader("Importance approximative des variables (permutation locale)")
    # importance locale autour de la séquence actuelle
    base_seq = seq.copy()
    base_pred = model.predict(base_seq)[0][0]
    diffs = {}

    for i, col in enumerate(FEATURES):
        seq_pert = base_seq.copy()
        # remplace la feature i dans toute la séquence par sa moyenne globale
        seq_pert[0, :, i] = np.mean(X_scaled[:, i])
        pred_pert = model.predict(seq_pert)[0][0]
        diffs[col] = abs(pred_pert - base_pred)

    imp_df = pd.DataFrame(
        {"feature": list(diffs.keys()), "impact (|Δsortie|)": list(diffs.values())}
    ).sort_values("impact (|Δsortie|)", ascending=False)

    st.bar_chart(imp_df.set_index("feature"))

    with st.expander("Données brutes pour l'explication"):
        st.write(imp_df)
