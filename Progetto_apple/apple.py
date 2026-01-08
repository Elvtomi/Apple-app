import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config("🍎 Apple Quality ML", layout="wide")

# ======================
# Sidebar con immagine + overlay leggibile
# ======================
sidebar_css = """
<style>
/* Sfondo sidebar */
[data-testid="stSidebar"] {
    background-image: url('https://cdn.pixabay.com/photo/2023/08/15/17/33/apples-8192411_1280.jpg');
    background-size: cover;
    background-position: center;
}

/* Overlay semi-trasparente solo dietro menu */
[data-testid="stSidebar"] > div:first-child {
    background-color: rgba(50, 50, 50, 0.5);
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
}

/* Testo leggibile */
[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: bold;
}

/* Radio button hover */
[data-testid="stSidebar"] input[type="radio"] + label {
    background-color: rgba(255,255,255,0.2) !important;
    border-radius: 5px;
    padding: 5px;
}
</style>
"""
st.markdown(sidebar_css, unsafe_allow_html=True)

# ======================
# Sidebar menu
# ======================
st.sidebar.title("🍎 Apple ML")
menu = st.sidebar.radio(
    "Menu",
    ["Carica Dataset", "EDA", "Inferenza"]
)

# ======================
# Load pipeline e modelli
# ======================
@st.cache_resource
def load_clean_pipeline():
    return joblib.load("models/pulizia_dati.pkl")

@st.cache_resource
def load_models():
    return {
        "Random Forest": joblib.load("models/rf_model.pkl"),
        "SVC": joblib.load("models/svc_model.pkl")
    }

# ======================
# CARICAMENTO DATASET + PULIZIA
# ======================
if menu == "Carica Dataset":
    st.header("📂 Carica Dataset")
    file = st.file_uploader("CSV o XLSX", ["csv", "xlsx"])
    if file:
        # Caricamento file
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.write("📄 Dataset originale")
        st.dataframe(df.head())

        # --------------------
        # Pulizia dati direttamente nel main
        # --------------------
        df_clean = df.copy()

        # Colonne da eliminare (NON usate nel training)
        cols_to_drop = ["Weight", "Acidity", "Crunchiness", "A_id"]

        df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

        # Rimuovi righe con NaN
        df_clean = df_clean.dropna()

        st.success("✅ Dati puliti")
        st.dataframe(df_clean.head())

        # Salva nel session_state
        st.session_state["data"] = df_clean

# ======================
# EDA
# ======================
elif menu == "EDA":
    st.header("📊 Analisi Esplorativa")
    if "data" not in st.session_state or st.session_state["data"].empty:
        st.warning("Carica prima un dataset nella sezione 'Carica Dataset'")
    else:
        df = st.session_state["data"]

        # Distribuzione target
        st.subheader("Distribuzione Target")
        target_col = "Quality" if "Quality" in df.columns else st.selectbox("Seleziona colonna target", df.columns)
        plt.figure(figsize=(3,2))
        df[target_col].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title(f"Distribuzione di {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
        plt.clf()

        # Distribuzioni variabili numeriche
        st.subheader("Distribuzioni Variabili Numeriche")
        for col in df.select_dtypes("number").columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax, color="tomato")
            ax.set_title(f"Distribuzione di {col}")
            st.pyplot(fig)

        # Matrice di correlazione
        st.subheader("Matrice di Correlazione")
        num_df = df.select_dtypes("number")
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# ======================
# INFERENZA
# ======================
elif menu == "Inferenza":
    st.header("🔮 Inferenza")
    
    if "data" not in st.session_state:
        st.warning("Carica prima un dataset")
    else:
        df = st.session_state["data"].copy()
        models = load_models()
        results = df.copy()

        # --------------------------
        # Seleziona solo le feature usate in training
        # --------------------------
        feature_cols = ['Size', 'Sweetness', 'Juiciness', 'Ripeness']
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Errore: mancano le colonne {missing} necessarie per il modello")
        else:
            X = df[feature_cols]

            # Mappatura predizioni numeriche -> testuali
            num_to_label = {0: "bad", 1: "good"}

            for name, model in models.items():
                # predizioni numeriche
                preds = model.predict(X)

                # Converti predizioni numeriche in label testuali
                preds_labels = [num_to_label[p] for p in preds]
                results[f"Prediction_{name}"] = preds_labels

                # --------------------------
                # Confusion Matrix solo se target presente
                # --------------------------
                if "Quality" in df.columns:
                    y_true = df["Quality"]

                    # Gestione target testuale -> numerico
                    if y_true.dtype == object:
                        label_to_num = {"bad": 0, "good": 1}
                        y_true_numeric = y_true.map(label_to_num)
                    else:
                        y_true_numeric = y_true

                    cm = confusion_matrix(y_true_numeric, preds)
                    cm_labels = ["bad", "good"]

                    # Visualizza confusion matrix
                    st.subheader(f"Confusion Matrix - {name}")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

            # --------------------------
            # Mostra risultati e download CSV
            # --------------------------
            st.subheader("📄 Predizioni")
            st.dataframe(results.head())

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predizioni",
                csv,
                file_name="predizioni.csv",
                mime="text/csv"
            )
