# Traffic Volume Predictor — Streamlit App

# Disclosure (for notebook/script grading only, not shown in app UI):
# I used ChatGPT minimally for:
#   • MAPIE prefit usage and alpha slider wiring
#   • small Streamlit layout details (tabs/expanders)
#   • quick syntax checks
# I verified and edited all code; modeling choices follow course materials.

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="Traffic Volume Predictor", layout="wide")

# ---------- Paths & helpers ----------
APP_DIR = Path(__file__).resolve().parent

def p(file):  # relative path helper
    return str(APP_DIR / file)

def file_exists(fname: str) -> bool:
    try:
        return (APP_DIR / fname).exists()
    except Exception:
        return False

# Load artifacts if available
mapie_path = APP_DIR / "mapie_traffic.pickle"
xgb_path   = APP_DIR / "xgb_traffic.pickle"

MAPIE = None
PIPE_ALL = None
if mapie_path.exists():
    with open(mapie_path, "rb") as f:
        MAPIE = pickle.load(f)
if xgb_path.exists():
    with open(xgb_path, "rb") as f:
        PIPE_ALL = pickle.load(f)

# Sample user CSV for schema preview
SAMPLE_DF = None
try:
    SAMPLE_DF = pd.read_csv(p("traffic_data_user.csv")).head(5)
except Exception:
    pass

# ---------- Sidebar ----------
with st.sidebar:
    st.image(p("traffic_sidebar.jpg"))
    st.caption("Traffic Volume Predictor")

    st.markdown("### Input Features")
    st.write("You can either upload your data file or manually enter input features.")

    exp_csv = st.expander("Option 1: Upload CSV File", expanded=False)
    exp_form = st.expander("Option 2: Fill Out Form", expanded=False)

# ---------- Title / Header area ----------
st.markdown(
    "<h1 style='text-align:center; color:#d4f542;'>Traffic Volume Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Utilize our advanced Machine Learning application to predict traffic volume.</p>",
    unsafe_allow_html=True,
)
st.image(p("traffic_image.gif"))

# Alpha slider (significance level)
alpha = st.slider(
    "Select alpha value for prediction intervals",
    min_value=0.01, max_value=0.50, value=0.10, step=0.01
)

# Hold user input (either CSV or form)
user_df = None
input_method = None

# ---------- CSV option ----------
with exp_csv:
    csv_file = st.file_uploader("Upload a CSV file containing traffic details.", type=["csv"])
    if SAMPLE_DF is not None:
        st.write("Sample Data Format for Upload")
        st.dataframe(SAMPLE_DF, use_container_width=True)
        st.warning("Ensure your uploaded file has the same column names and data types as shown above.")

    if csv_file is not None:
        try:
            df_up = pd.read_csv(csv_file)
            user_df = df_up.copy()
            input_method = "csv"
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# Form option
with exp_form:
    default_cols = [
        "holiday", "temp", "rain_1h", "snow_1h",
        "clouds_all", "weather_main", "month", "weekday", "hour"
    ]
    guide_cols = list(SAMPLE_DF.columns) if SAMPLE_DF is not None else default_cols

    # Basic choices from dataset semantics
    holiday_opt = ["None", "Labor Day", "Christmas Day", "Veterans Day", "Thanksgiving Day",
                   "Independence Day", "New Years Day", "Washingtons Birthday", "Memorial Day",
                   "Columbus Day", "Martin Luther King Jr Day"]
    weather_opt = ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain",
                   "Smoke", "Snow", "Squall", "Thunderstorm"]
    month_opt = ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"]
    weekday_opt = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    hour_opt = list(range(0, 24))

    # Build a single-row dict using typical field names; tolerate missing columns
    c1, c2 = st.columns(2)
    with c1:
        f_holiday = st.selectbox("Choose whether today is a designated holiday or not", holiday_opt, index=0)
        f_temp    = st.number_input("Average temperature in Kelvin", value=281.21, step=0.25)
        f_rain    = st.number_input("Amount in mm of rain that occurred in the hour", value=0.33, step=0.01)
        f_snow    = st.number_input("Amount in mm of snow that occurred in the hour", value=0.00, step=0.01)
    with c2:
        f_clouds  = st.number_input("Percentage of cloud cover", value=49, step=1, min_value=0, max_value=100)
        f_weather = st.selectbox("Choose the current weather", weather_opt, index=1)
        f_month   = st.selectbox("Choose month", month_opt, index=0)
        f_weekday = st.selectbox("Choose day of the week", weekday_opt, index=0)
        f_hour    = st.selectbox("Choose hour", hour_opt, index=5)

    if st.button("Submit Form Data"):
        # Construct single row using whatever columns the model expects
        row = {}
        for col in guide_cols:
            cl = col.lower()
            if cl in ("holiday", "_holiday"):
                row[col] = f_holiday
            elif cl in ("temp", "temperature", "temperature_k", "avg_temp_k"):
                row[col] = f_temp
            elif cl in ("rain_1h", "rain"):
                row[col] = f_rain
            elif cl in ("snow_1h", "snow"):
                row[col] = f_snow
            elif cl in ("clouds_all", "clouds", "cloud_cover"):
                row[col] = f_clouds
            elif cl in ("weather_main", "weather"):
                row[col] = f_weather
            elif cl in ("month",):
                row[col] = f_month
            elif cl in ("weekday", "dayofweek", "day_of_week"):
                row[col] = f_weekday
            elif cl in ("hour", "hr"):
                row[col] = f_hour
            else:
                # default blank for any extra columns present in the training schema
                row[col] = None

        user_df = pd.DataFrame([row])
        input_method = "form"

# Prediction area
st.markdown("## Predicting Traffic Volume...")

if MAPIE is None:
    st.warning("Model artifacts not found. Please ensure 'mapie_traffic.pickle' is in the app folder.")
elif user_df is None:
    st.info("Please choose a data input method to proceed.")
else:
    try:
        # Run predictions + intervals
        y_pred, y_pis = MAPIE.predict(user_df, alpha=alpha)
        user_pred = np.asarray(y_pred).ravel()
        lo = y_pis[:, 0, 0]
        hi = y_pis[:, 1, 0]

        if input_method == "form":
            # Single prediction display
            st.subheader("Predicted Traffic Volume")
            st.metric(label="Predicted Traffic Volume", value=f"{int(round(user_pred[0]))}")
            st.caption(f"Prediction Interval ({int((1-alpha)*100)}%): [{int(round(lo[0]))}, {int(round(hi[0]))}]")
        else:
            # CSV mode: append columns & show
            out = user_df.copy()
            out["Predicted Volume"] = np.round(user_pred, 2)
            out["Lower PI"] = np.round(lo, 2)
            out["Upper PI"] = np.round(hi, 2)
            st.dataframe(out, use_container_width=True)
            # Optional: allow download
            st.download_button(
                "Download predictions as CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="traffic_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Model performance + plots
st.markdown("## Model Performance and Inference")
tabs = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs. Actual", "Coverage Plot"])

with tabs[0]:
    st.write("### Feature Importance")
    fig_file = "feature_importance.svg"
    if file_exists(fig_file):
        st.image(p("traffic_image.gif"))
    else:
        st.caption("Feature importance plot not found. Re-run the notebook to generate it.")

with tabs[1]:
    st.write("### Histogram of Residuals")
    fig_file = "residual_hist.svg"
    if file_exists(fig_file):
        st.image(p(fig_file))
    else:
        st.caption("Residuals histogram not found. Re-run the notebook to generate it.")

with tabs[2]:
    st.write("### Predicted vs. Actual")
    fig_file = "pred_vs_actual.svg"
    if file_exists(fig_file):
        st.image(p(fig_file))
    else:
        st.caption("Predicted vs. Actual plot not found. Re-run the notebook to generate it.")

with tabs[3]:
    st.write("### Coverage Plot")
    fig_file = "coverage_plot.svg"
    if file_exists(fig_file):
        st.image(p(fig_file))
    else:
        st.caption("Coverage plot not found. Re-run the notebook to generate it.")
