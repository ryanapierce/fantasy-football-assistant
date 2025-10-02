import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from joblib import dump, load


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
DATA_PATH = ROOT.parent / "fantasy-football-assistant" / "final_1_lag_ffa_dataset.csv"


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric identifier columns and keep only _r_avg_1 rolling features plus season.
    Mirrors the notebook's logic for model input.
    """
    # Keep rolling average 1-window features and season
    keep = [c for c in df.columns if "_r_avg_1" in c or c == "season"]

    # Also keep any numeric columns that are already numbers (safe fallback)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = list(dict.fromkeys(keep + numeric_cols))

    # Drop object/id/name/team columns unless explicitly in keep
    drop_cols = [c for c in df.columns if (df[c].dtype == object or "id" in c or "name" in c or "team" in c) and c not in keep]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Ensure we return only kept columns (except target if present)
    return df.loc[:, [c for c in df.columns if c in keep or df[c].dtype != object]]


def build_pipeline(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # simple numeric pipeline
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])

    pre = ColumnTransformer([("num", num_pipe, num_cols)])

    pipe = Pipeline([("pre", pre), ("est", Ridge(alpha=1.0))])
    return pipe


def train_and_save_model(csv_path: Path, model_path: Path):
    df = pd.read_csv(csv_path)
    # target we'll predict: fantasy_points_r_avg_1 if available, else fantasy_points
    if "fantasy_points_r_avg_1" in df.columns:
        target = "fantasy_points_r_avg_1"
    elif "fantasy_points" in df.columns:
        target = "fantasy_points"
    else:
        raise RuntimeError("No target column found in dataset. Add 'fantasy_points_r_avg_1' or 'fantasy_points'.")

    df = preprocess_df(df)
    if target not in df.columns:
        # If preprocessing dropped the target, reload original and extract target
        orig = pd.read_csv(csv_path)
        df[target] = orig[target]

    X = df.drop(columns=[target])
    y = df[target]

    pipe = build_pipeline(X)
    pipe.fit(X, y)
    dump(pipe, model_path)
    return pipe


def load_model(model_path: Path):
    if model_path.exists():
        return load(model_path)
    return None


def predict_dataframe(model, df: pd.DataFrame) -> pd.Series:
    df_proc = preprocess_df(df)
    # align columns with model's training columns: ColumnTransformer will handle missing cols but warn
    return pd.Series(model.predict(df_proc), index=df.index)


def main():
    st.title("Fantasy Football Draft Assistant — Linear Regression Predictor")

    st.sidebar.markdown("## Model")
    model = load_model(MODEL_PATH)

    if model is None:
        st.sidebar.write("No trained model found.")
        if st.sidebar.button("Train model from local dataset"):
            if DATA_PATH.exists():
                with st.spinner("Training model — this may take a few seconds..."):
                    try:
                        model = train_and_save_model(DATA_PATH, MODEL_PATH)
                        st.sidebar.success(f"Model trained and saved to {MODEL_PATH}")
                    except Exception as e:
                        st.sidebar.error(f"Training failed: {e}")
            else:
                st.sidebar.error(f"Training dataset not found at {DATA_PATH}")
    else:
        st.sidebar.success(f"Loaded model from {MODEL_PATH}")

    st.sidebar.markdown("---")

    st.header("Predict from CSV")
    uploaded = st.file_uploader("Upload a CSV with player features", type=["csv"])
    if uploaded is not None and model is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Run predictions on uploaded CSV"):
            try:
                preds = predict_dataframe(model, df)
                out = df.copy()
                out["predicted_fantasy_points"] = preds
                st.write(out.head())
                csv = out.to_csv(index=False)
                st.download_button("Download predictions CSV", csv, file_name="predictions.csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.header("Quick single prediction")
    st.markdown("Enter numeric feature values for a quick single-row prediction. Only numeric columns preserved in training are required.")
    if model is not None:
        # get numeric feature names from model pipeline
        try:
            # best-effort: inspect column names from the training data by loading DATA_PATH
            sample = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
            sample = preprocess_df(sample)
            feature_cols = [c for c in sample.columns if c != "fantasy_points_r_avg_1" and c != "fantasy_points"]
        except Exception:
            feature_cols = []

        inputs = {}
        for c in feature_cols:
            inputs[c] = st.number_input(c, value=float(sample[c].dropna().mean()) if c in sample.columns else 0.0)

        if st.button("Predict single row"):
            if not feature_cols:
                st.error("No feature columns detected to build a single-row input.")
            else:
                row = pd.DataFrame([inputs])
                try:
                    pred = model.predict(row)[0]
                    st.write(f"Predicted fantasy points: {pred:.3f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    # If the user executes this file with plain `python streamlit_app.py`, Streamlit
    # will try to run UI calls without a ScriptRunContext and emit the warning:
    # "missing ScriptRunContext! This warning can be ignored when running in bare mode.".
    #
    # Prefer running the app with the Streamlit CLI. Detect common Streamlit
    # environment variables to decide whether we're running under `streamlit run`.
    import os
    import sys

    streamlit_env_vars = [
        "STREAMLIT_RUN_MAIN",
        "STREAMLIT_SERVER_PORT",
        "STREAMLIT_SERVER_RUNNING",
        "STREAMLIT_SERVER_HEADLESS",
    ]

    if not any(os.environ.get(v) for v in streamlit_env_vars):
        print("This file is a Streamlit app. Run it with:\n\n  streamlit run streamlit_app.py\n")
        sys.exit(0)

    main()
