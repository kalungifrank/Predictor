"""
Inhibitor Predictor
--------------------
A Streamlit app that takes a dataset of molecules (SMILES + activity label),
computes molecular fingerprints, trains scikit-learn classifiers, reports
performance, and predicts activity for a new set of molecules.

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay,
)

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

st.set_page_config(page_title="Inhibitor Predictor", layout="wide")

# ----------------------------------------------------------------------
# Embedded sample data (so there is always a one-click way to try the app)
# ----------------------------------------------------------------------

SAMPLE_TRAINING_CSV = """smiles,activity
CC(=O)OC1=CC=CC=C1C(=O)O,active
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,active
CC(C)Cc1ccc(cc1)C(C)C(=O)O,inactive
CC(=O)Nc1ccc(O)cc1,active
c1ccccc1,inactive
CCO,inactive
CC(=O)Oc1ccccc1C(=O)O,active
Clc1ccc(cc1)C(c1ccccc1)C(Cl)(Cl)Cl,inactive
COc1cc2c(cc1OC)C(=O)C(CC2)Cc1ccc(OC)c(OC)c1,active
O=C(O)c1ccccc1O,active
CC1=CC(=O)CC(C)(C)C1,inactive
CC(C)(C)NCC(O)c1ccc(O)c(CO)c1,active
Nc1ccc(cc1)S(=O)(=O)N,active
CC(=O)Nc1ccc(cc1)O,active
Clc1ccccc1,inactive
CCN(CC)CCOC(=O)c1ccc(N)cc1,active
CC(C)NCC(O)COc1cccc2ccccc12,active
Oc1ccccc1,inactive
CC(C)Cc1ccc(cc1)C(C)C(N)=O,inactive
CN(C)CCOC(c1ccccc1)c1ccccc1,active
"""

SAMPLE_PREDICT_CSV = """compound_id,smiles
NC001,CC(=O)Oc1ccccc1C(=O)O
NC002,CC(C)Cc1ccc(cc1)C(C)C(=O)O
NC003,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
NC004,c1ccc2ccccc2c1
NC005,CC(=O)Nc1ccc(O)cc1
NC006,CCOC(=O)c1ccccc1N
NC007,Oc1ccc(cc1)C(=O)O
NC008,CC(C)(C)NCC(O)c1ccc(O)c(CO)c1
NC009,Clc1ccc(Cl)cc1
NC010,CN(C)CCOC(c1ccccc1)c1ccccc1
"""

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

FP_OPTIONS = ["Morgan (ECFP4)", "Morgan (ECFP6)", "MACCS Keys", "RDKit Fingerprint"]

MODEL_OPTIONS = {
    "Random Forest": RandomForestClassifier,
    "Support Vector Machine (SVM)": SVC,
    "Logistic Regression": LogisticRegression,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
}


def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol
    except Exception:
        return None


def compute_fingerprint(mol, fp_type):
    if mol is None:
        return None
    if fp_type == "Morgan (ECFP4)":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    elif fp_type == "Morgan (ECFP6)":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
    elif fp_type == "MACCS Keys":
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "RDKit Fingerprint":
        fp = Chem.RDKFingerprint(mol)
    else:
        return None
    arr = np.zeros((fp.GetNumBits(),), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_dataframe(df, smiles_col, fp_type, progress_label="Computing fingerprints"):
    fps, valid_idx, invalid_rows = [], [], []
    progress = st.progress(0, text=progress_label)
    n = len(df)
    for i, smi in enumerate(df[smiles_col].tolist()):
        mol = smiles_to_mol(smi)
        if mol is None:
            invalid_rows.append(i)
        else:
            fp = compute_fingerprint(mol, fp_type)
            fps.append(fp)
            valid_idx.append(i)
        if n > 0:
            progress.progress((i + 1) / n, text=progress_label)
    progress.empty()
    X = np.array(fps) if fps else np.empty((0, 0))
    return X, valid_idx, invalid_rows


def get_model(name, params):
    cls = MODEL_OPTIONS[name]
    return cls(**params)


def model_param_widgets(name):
    params = {}
    if name == "Random Forest":
        params["n_estimators"] = st.slider("n_estimators", 50, 500, 200, 50)
        params["max_depth"] = st.select_slider("max_depth", options=[None, 5, 10, 20, 30], value=None)
        params["random_state"] = 42
    elif name == "Support Vector Machine (SVM)":
        params["C"] = st.slider("C", 0.01, 10.0, 1.0)
        params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"])
        params["probability"] = True
        params["random_state"] = 42
    elif name == "Logistic Regression":
        params["C"] = st.slider("C", 0.01, 10.0, 1.0)
        params["max_iter"] = 2000
        params["random_state"] = 42
    elif name == "K-Nearest Neighbors":
        params["n_neighbors"] = st.slider("n_neighbors", 1, 30, 5)
    elif name == "Gradient Boosting":
        params["n_estimators"] = st.slider("n_estimators", 50, 500, 150, 50)
        params["learning_rate"] = st.select_slider(
            "learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1
        )
        params["random_state"] = 42
    return params


def to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ----------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------
for key in ["raw_df", "X", "y", "valid_idx", "fp_type", "smiles_col", "label_col",
            "model", "scaler", "feature_names", "X_test", "y_test", "y_pred", "y_proba",
            "classes_", "used_sample", "new_df_sample"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("🧪 Inhibitor Predictor")
st.caption(
    "Upload a labeled molecule dataset, compute fingerprints, train a classifier, "
    "review performance, then predict activity for new compounds."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Upload Data", "2. Fingerprints", "3. Train & Evaluate", "4. Predict New Compounds"]
)

# ----------------------------------------------------------------------
# TAB 1 — Upload
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Upload training dataset")
    st.write(
        "CSV file with at least one column of SMILES strings and one column with the "
        "binary activity label (e.g. `active`/`inactive` or `1`/`0`)."
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        file = st.file_uploader("Training CSV", type=["csv"], key="train_upload")
    with c2:
        st.write("")
        st.write("")
        if st.button("🧪 Try with sample dataset", use_container_width=True):
            df = pd.read_csv(io.StringIO(SAMPLE_TRAINING_CSV))
            st.session_state.raw_df = df
            st.session_state.used_sample = True

    if file is not None:
        df = pd.read_csv(file)
        st.session_state.raw_df = df
        st.session_state.used_sample = False

    if st.session_state.raw_df is not None:
        df_preview = st.session_state.raw_df
        if st.session_state.get("used_sample"):
            st.info(
                "Using the built-in sample dataset (20 small molecules, labeled "
                "active/inactive) — just to help you explore the workflow. "
                "Upload your own CSV above to replace it."
            )
        st.success(f"Loaded {len(df_preview)} rows, {len(df_preview.columns)} columns.")
        st.dataframe(df_preview.head(10), use_container_width=True)

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            smiles_col = st.selectbox("SMILES column", cols, key="smiles_col_select")
        with c2:
            label_col = st.selectbox(
                "Activity / label column",
                cols,
                index=min(1, len(cols) - 1),
                key="label_col_select",
            )
        st.session_state.smiles_col = smiles_col
        st.session_state.label_col = label_col

        if df[label_col].nunique() > 10:
            st.warning(
                "Selected label column has many unique values — make sure it's a "
                "classification target (e.g. active/inactive), not a continuous value."
            )
        else:
            st.info(f"Class distribution:\n\n{df[label_col].value_counts().to_string()}")

# ----------------------------------------------------------------------
# TAB 2 — Fingerprints
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Compute molecular fingerprints")
    if st.session_state.raw_df is None:
        st.info("Upload a dataset in Tab 1 first.")
    else:
        fp_type = st.selectbox("Fingerprint type", FP_OPTIONS, key="fp_type_select")

        if st.button("Compute fingerprints", type="primary"):
            df = st.session_state.raw_df
            X, valid_idx, invalid_rows = featurize_dataframe(
                df, st.session_state.smiles_col, fp_type
            )
            if len(invalid_rows) > 0:
                st.warning(
                    f"{len(invalid_rows)} row(s) had unparsable SMILES and were dropped "
                    f"(rows: {invalid_rows[:20]}{'...' if len(invalid_rows) > 20 else ''})."
                )
            y = df[st.session_state.label_col].to_numpy(dtype=object)[valid_idx]
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.valid_idx = valid_idx
            st.session_state.fp_type = fp_type
            st.success(f"Computed {fp_type} fingerprints: X shape = {X.shape}")

        if st.session_state.X is not None:
            st.write("Feature matrix preview (first 5 rows, first 20 bits):")
            st.dataframe(pd.DataFrame(st.session_state.X[:5, :20]), use_container_width=True)
            st.write(f"**Shape:** {st.session_state.X.shape[0]} molecules × "
                     f"{st.session_state.X.shape[1]} bits — fingerprint type: "
                     f"**{st.session_state.fp_type}**")

# ----------------------------------------------------------------------
# TAB 3 — Train & Evaluate
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Train a classifier")
    if st.session_state.X is None:
        st.info("Compute fingerprints in Tab 2 first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        with c2:
            test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2, 0.05)
        with c3:
            scale_features = st.checkbox("Standardize features", value=False)

        with st.expander("Model hyperparameters"):
            params = model_param_widgets(model_name)

        if st.button("Train model", type="primary"):
            X, y = st.session_state.X, st.session_state.y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            scaler = None
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model = get_model(model_name, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.y_proba = y_proba
            st.session_state.classes_ = model.classes_
            st.success("Model trained.")

        if st.session_state.model is not None:
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            acc = accuracy_score(y_test, y_pred)

            st.metric("Accuracy", f"{acc:.3f}")

            st.write("**Classification report**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

            colA, colB = st.columns(2)
            with colA:
                st.write("**Confusion matrix**")
                cm = confusion_matrix(y_test, y_pred, labels=st.session_state.classes_)
                fig, ax = plt.subplots()
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=st.session_state.classes_,
                    yticklabels=st.session_state.classes_, ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with colB:
                classes = st.session_state.classes_
                if st.session_state.y_proba is not None and len(classes) == 2:
                    st.write("**ROC curve**")
                    pos_label = classes[1]
                    y_test_bin = (np.array(y_test) == pos_label).astype(int)
                    scores = st.session_state.y_proba[:, 1]
                    fpr, tpr, _ = roc_curve(y_test_bin, scores)
                    roc_auc = auc(fpr, tpr)
                    fig2, ax2 = plt.subplots()
                    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax2)
                    st.pyplot(fig2)
                else:
                    st.info("ROC curve shown for binary classifiers with probability output.")

            if hasattr(st.session_state.model, "feature_importances_"):
                st.write("**Top 20 most important fingerprint bits**")
                importances = st.session_state.model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:20]
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                ax3.barh(
                    [f"bit {i}" for i in top_idx][::-1],
                    importances[top_idx][::-1],
                )
                ax3.set_xlabel("Importance")
                st.pyplot(fig3)

# ----------------------------------------------------------------------
# TAB 4 — Predict new compounds
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Predict on new compounds")
    if st.session_state.model is None:
        st.info("Train a model in Tab 3 first.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            new_file = st.file_uploader("New compounds CSV", type=["csv"], key="predict_upload")
        with c2:
            st.write("")
            st.write("")
            if st.button("🧪 Try with sample compounds", use_container_width=True):
                st.session_state.new_df_sample = pd.read_csv(io.StringIO(SAMPLE_PREDICT_CSV))

        new_df = None
        if new_file is not None:
            new_df = pd.read_csv(new_file)
        elif st.session_state.get("new_df_sample") is not None:
            new_df = st.session_state.new_df_sample
            st.info("Using the built-in sample compound list (10 molecules, no labels).")

        if new_df is not None:
            st.dataframe(new_df.head(10), use_container_width=True)
            pred_smiles_col = st.selectbox(
                "SMILES column in new file", new_df.columns.tolist(), key="pred_smiles_col"
            )

            if st.button("Run predictions", type="primary"):
                X_new, valid_idx, invalid_rows = featurize_dataframe(
                    new_df, pred_smiles_col, st.session_state.fp_type,
                    progress_label="Featurizing new compounds",
                )
                if len(invalid_rows) > 0:
                    st.warning(
                        f"{len(invalid_rows)} row(s) had unparsable SMILES and were skipped."
                    )
                if X_new.shape[0] == 0:
                    st.error("No valid molecules to predict on.")
                else:
                    if st.session_state.scaler is not None:
                        X_new = st.session_state.scaler.transform(X_new)

                    model = st.session_state.model
                    preds = model.predict(X_new)
                    result_df = new_df.iloc[valid_idx].copy()
                    result_df["predicted_label"] = preds

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_new)
                        for i, cls in enumerate(model.classes_):
                            result_df[f"proba_{cls}"] = proba[:, i]

                    st.success(f"Predicted {len(result_df)} compounds.")
                    st.dataframe(result_df, use_container_width=True)

                    st.download_button(
                        "Download predictions as CSV",
                        data=to_csv_bytes(result_df),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                    st.write("**Predicted class distribution**")
                    fig4, ax4 = plt.subplots()
                    result_df["predicted_label"].value_counts().plot(kind="bar", ax=ax4)
                    ax4.set_ylabel("Count")
                    st.pyplot(fig4)

st.sidebar.header("About")
st.sidebar.write(
    "This app predicts whether new molecules are likely to be active inhibitors, "
    "based on molecular fingerprints and a model trained on your labeled dataset.\n\n"
    "**Workflow:** Upload → Fingerprints → Train & Evaluate → Predict"
)
st.sidebar.write(
    "**Input format:** CSV with a SMILES column and a binary activity/label column "
    "(e.g. `active`/`inactive`)."
)