import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==============================
# 📁 Directory Setup
# ==============================
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "outputs" / "plots"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "xgb_best_model.pkl"

# ==============================
# 📊 Load Dataset
# ==============================
def load_data():
    filepath = DATA_DIR / "features_rossmann.parquet"
    st.write(f"📁 Loading data from: {filepath}")
    return pd.read_parquet(filepath, engine="fastparquet")

df = load_data()

# ==============================
# 📌 Tabs
# ==============================
st.set_page_config(layout="wide")
st.title("🛍️ Retail Demand Forecasting Dashboard")

# ------------------------------
# 📊 Dataset Overview
# ------------------------------
st.header("📊 Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head(10))

# ------------------------------
# 📈 Forecast Results
# ------------------------------
st.header("📈 Forecast Results")

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

    features_path = MODEL_DIR / "xgb_best_model_features.pkl"
    if features_path.exists():
        expected_features = joblib.load(features_path)
    else:
        st.error("❌ Feature list not found. Please retrain model with feature list saved.")
        st.stop()

    # Check missing features
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        st.error(f"❌ Missing expected features: {missing}")
    else:
        X = df[expected_features].copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype("category")

        y_true = df["Sales"]
        y_pred = model.predict(X)

        plot_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        if len(plot_df) > 1000:
            plot_df = plot_df.tail(1000)

        st.line_chart(plot_df)

else:
    st.warning(f"Model file not found at {MODEL_PATH}. Please train and save the model first.")

# ------------------------------
# 🔍 SHAP Explainability
# ------------------------------
st.header("🔍 SHAP Explainability")

shap_plot = PLOT_DIR / "shap_xgb_best_model.png"
shap_bar = PLOT_DIR / "shap_bar_xgb_best_model.png"

if shap_plot.exists():
    st.subheader("SHAP Summary Plot")
    st.image(str(shap_plot), width="stretch")
else:
    st.info("SHAP summary plot not found.")

if shap_bar.exists():
    st.subheader("SHAP Bar Plot")
    st.image(str(shap_bar), width="stretch")
else:
    st.info("SHAP bar plot not found.")

# ------------------------------
# 💰 Price Simulation
# ------------------------------
st.header("🧪 Price Simulation")
st.info("This section simulates the impact of price changes (±10%, ±20%) on predicted sales.")

with st.expander("📉 Price Simulation – Impact on Predicted Sales", expanded=True):
    st.subheader("🧪 Forecast Simulation Based on Price Changes")

    # Load the simulation results
    sim_path = BASE_DIR / "outputs" / "price_simulation_predictions.csv"
    try:
        sim_df = pd.read_csv(sim_path, parse_dates=True, index_col=0)
        st.success("✅ Simulation results loaded.")
    except Exception as e:
        st.error(f"Error loading simulation data: {e}")
        st.stop()

    # Limit to last 200 rows
    sim_plot_df = sim_df.tail(200)

    # Line plot of variants
    st.write("### Simulated Sales Under Price Change Scenarios")
    st.write("Model was trained **without Price** to avoid leakage. Price was added later for simulation only.")

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in sim_plot_df.columns:
        ax.plot(sim_plot_df.index, sim_plot_df[col], label=col)
    ax.set_title("Impact of Price Changes on Predicted Sales (Last 200 Days)")
    ax.set_xlabel("Date Index")
    ax.set_ylabel("Predicted Sales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 📥 Download CSV
    csv = sim_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="📥 Download Simulation CSV",
        data=csv,
        file_name="price_simulation_predictions.csv",
        mime="text/csv",
    )

    st.info("💡 Simulated price elasticity using ±20%, ±10%, and baseline 0% change. Useful for pricing strategy planning.")
# ------------------------------
# 🎚️ Interactive Price Slider Simulation
# ------------------------------
with st.expander("🎛️ Interactive Price Slider Simulation", expanded=False):
    st.subheader("🎯 Adjust Price and View Predicted Sales Impact")

    # Load model + features again (if not already)
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        expected_features = joblib.load(MODEL_DIR / "xgb_best_model_features.pkl")
    else:
        st.error("❌ Model not found.")
        st.stop()

    # Slider for price change
    price_change = st.slider("Select Price Change (%)", -30, 30, 0, step=5)

    # Recreate base input without leakage (no Price in training features)
    X_base = df[expected_features].copy()

    # ✅ FIX: Convert object → category (MISSING EARLIER)
    for col in X_base.columns:
        if X_base[col].dtype == 'object':
            X_base[col] = X_base[col].astype("category")

    # Simulate sales change based on assumed elasticity
    elasticity = 0.4
    adjustment_factor = 1 - (price_change / 100) * elasticity

    y_pred_base = model.predict(X_base)
    y_pred_adjusted = y_pred_base * adjustment_factor

    # Build plot DataFrame
    plot_df = pd.DataFrame({
        "Original Forecast": y_pred_base,
        f"Adjusted Forecast ({price_change:+}%)": y_pred_adjusted
    })

    # Plot last 300 days
    st.line_chart(plot_df.tail(300))

    st.caption(f"🧪 Assumed elasticity = {elasticity}. This means for every 1% price increase, sales drop by 0.4%.")
